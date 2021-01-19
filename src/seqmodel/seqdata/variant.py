import sys
sys.path.append('./src')
import torch

from seqmodel import BASE_TO_INDEX, N_BASE, INDEX_TO_BASE
from seqmodel.functional import bioseq_to_index


class MutationModel():

    def __init__(self):
        self.pmf = torch.zeros([3, 4])
        self._update_cmf()

    """
    PMF is a matrix where dim=0 rows are target, dim=1 cols are source
    3 mutations per nucleotide, permutation is row index + 1,
    e.g. row 0 is permute by 1, row 1 permute by 2, etc.
    """
    def _edit_pmf(self, source, target, prob):
        row = (BASE_TO_INDEX[target] - BASE_TO_INDEX[source]) % N_BASE - 1
        col = BASE_TO_INDEX[source]
        self.pmf[row, col] = prob

    def _update_cmf(self):
        self.cmf = torch.cumsum(self.pmf, dim=0)
        if torch.any(self.cmf < 0.) or torch.any(self.cmf > 1.):
            raise ValueError('Cumulative probability out of (0, 1) bounds: ', self.cmf)


class FixedRateSubstitutionModel(MutationModel):

    """
    Specify at least one of the following:
        global_rate: probability (float between 0. and 1.)
        transition_rate: overrides global_rate for transitions
        transversion_rate: overrides global_rate for transversions
        nucleotide_rates: dict of dicts (source to target, e.g. A->T is {'A':{'T': 0.1}})
            for nucleotides indexed as `transform.INDEX_TO_BASE`, overrides previous
    """
    def __init__(self,
                global_rate=None,
                transition_rate=None,
                transversion_rate=None,
                nucleotide_rates=None):
        self._ts_rate, self._tv_rate = 0., 0.
        if global_rate is not None:
            self._ts_rate = global_rate / 3  # 3 mutations possible per base
            self._tv_rate = global_rate / 3
        if transition_rate is not None:
            self._ts_rate = transition_rate
        if transversion_rate is not None:
            self._tv_rate = transversion_rate / 2  # 2 transversions per base
        
        self.pmf = torch.full([3, 4], self._tv_rate)
        # transitions affect A<->G and C<->T
        self._edit_pmf('G', 'A', self._ts_rate)
        self._edit_pmf('A', 'G', self._ts_rate)
        self._edit_pmf('T', 'C', self._ts_rate)
        self._edit_pmf('C', 'T', self._ts_rate)

        if nucleotide_rates is not None:
            for src, v in nucleotide_rates.items():
                for tgt, prob in v.items():
                    self._edit_pmf(src, tgt, prob)
        self._update_cmf()


"""
Function instead of object in order to be thread safe. Takes in MutationModel object
 which gives cumulative probabilities for each mutation type.
Sequence should contain bases in indexed form (i.e. tensor of dtype=torch.long).
    omit_indexes: list of sequence indexes (indexed relative to start of sequence)
        which should not have variants
    return: tuple of chrom, pos (1-indexed), ref, alt columns (arrays)
"""
def gen_variants(mutation_model, sequence, key, coord,
            omit_indexes=None, as_pyvcf_format=False, min_variants=0):
    uniform = torch.rand_like(sequence, dtype=torch.float)  # probabilities to compare to
    prob_cutoffs = torch.stack([torch.gather(row, -1, sequence) for row in mutation_model.cmf], dim=0)
    if omit_indexes is not None:  # zero probability of any mutation at omit_indexes
        prob_cutoffs[..., omit_indexes] = 0.
    cumulative = torch.sum(uniform < prob_cutoffs, dim=0).to(torch.long)
    indexes = torch.nonzero(cumulative, as_tuple=True)[0]
    print(indexes.shape)
    n_variants = indexes.size(-1)

    n_required = min_variants - n_variants
    if n_required > 0:  # generate more if less than min_variants
        valid_indexes = torch.ones([sequence.size(-1)])
        valid_indexes[omit_indexes] = 0.
        additional_indexes = torch.nonzero(valid_indexes, as_tuple=True)[0]
        selected = additional_indexes[torch.randperm(additional_indexes.size(-1))[n_required]]
        indexes = torch.cat([indexes, selected], dim=0)
        cumulative = torch.cat([cumulative, torch.randint(1, 3, [min_variants - n_variants])], dim=0)

    chrom = [key] * n_variants
    pos = indexes + coord + 1  # convert to 1-indexed
    ref = sequence[indexes]
    # subtract cumulative from sequence because of uniform < row comparison
    alt = torch.remainder(ref - cumulative[indexes], N_BASE)

    if as_pyvcf_format:
        return [PyVCFRecord(c, p.item(), r.item(), a.item()) for c, p, r, a in zip(chrom, pos, ref, alt)]
    return chrom, pos, ref, alt

"""
This is a temporary shortcut to allow variants to be treated like pyvcf _Record objects,
since pyvcf does not have API for creating them.
"""
class PyVCFRecord():

    def __init__(self, chrom, pos, ref, alt):
        self.CHROM = chrom
        self.POS = pos
        self.REF = INDEX_TO_BASE[ref]
        self.ALT = [self.PyVCFAltRecord(INDEX_TO_BASE[alt])]

    class PyVCFAltRecord():

        def __init__(self, alt):
            self.sequence = alt


"""
    sequence: tensor of indexed bases
    variants: table containg chrom (str), pos (int), ref (str), alt ([str]) from VCF
"""
def apply_variants(sequence, key, coord, variants):
    last_index = 0
    subseqs = []
    for row in variants:
        chrom, pos, ref, alt = row.CHROM, row.POS, row.REF, row.ALT
        pos -= 1  # convert to zero-indexed coordinates
        if key == chrom and pos >= coord and pos < coord + sequence.size(-1) \
                    and alt[0] is not None:  # ignore if variant is missing '.'
            var_pos = pos - coord
            # must match ref seq
            seq_diff = sequence[var_pos:var_pos + len(ref)] != bioseq_to_index(ref)
            if torch.any(seq_diff):
                raise ValueError('Reference sequence differs at', pos,
                        sequence[var_pos:var_pos + len(ref)], ref)
            subseqs.append(sequence[last_index:var_pos])
            # if more than one variant, use first one
            subseqs.append(bioseq_to_index(alt[0].sequence))
            last_index = var_pos + len(ref)
    subseqs.append(sequence[last_index:])  # remaining part of ref seq
    return torch.cat(subseqs, dim=-1)


def calc_mutation_rates(ref_seq, var_seq):
    variants = (ref_seq != var_seq)
    ref_allele = ref_seq[variants]
    var_allele = var_seq[variants]
    A = BASE_TO_INDEX['A']
    G = BASE_TO_INDEX['G']
    C = BASE_TO_INDEX['C']
    T = BASE_TO_INDEX['T']
    is_transition = torch.logical_or(torch.logical_or(
                    torch.logical_and(ref_seq == A, var_seq == G),
                    torch.logical_and(ref_seq == G, var_seq == A)),
                    torch.logical_or(torch.logical_and(ref_seq == C, var_seq == T),
                    torch.logical_and(ref_seq == T, var_seq == C)))
    is_transversion = torch.logical_and(ref_seq != var_seq,
                        torch.logical_not(is_transition))
    n_transition = torch.sum(is_transition.float()).item()
    n_transversion = torch.sum(is_transversion.float()).item()
    seq_len = ref_seq.size(-1)
    return n_transition / seq_len, n_transversion / seq_len


# TODO indels
