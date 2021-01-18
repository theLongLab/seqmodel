import sys
sys.path.append('./src')
import torch

from seqmodel import BASE_TO_INDEX, N_BASE, INDEX_TO_BASE
from seqmodel.functional import bioseq_to_index


"""
    sequence: tensor of indexed bases
    variants: table containg chrom (str), pos (int), ref (str), alt ([str]) from VCF
"""
def apply_variants(sequence, key, coord, variants):
    last_index = 0
    subseqs = []
    for row in variants:
        chrom, pos, ref, alt = row.CHROM, row.POS, row.REF, row.ALT
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


class MutationModel():

    """
    Sequence should contain bases in indexed form (i.e. tensor of dtype=torch.long).
        omit_positions: list of coordinates which should not have variants
        return: tuple of chrom, pos, ref, alt columns (arrays)
    """
    def gen_variants(self, sequence, key, coord, omit_positions=None):
        raise NotImplementedError()


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
        self.tt_prob = 0.
        self.tv_prob = 0.
        self.set_rates(global_rate=global_rate, transition_rate=transition_rate,
            transversion_rate=transversion_rate, nucleotide_rates=nucleotide_rates)
    
    def set_rates(self,
                global_rate=None,
                transition_rate=None,
                transversion_rate=None,
                nucleotide_rates=None):
        if global_rate is not None:
            self.tt_prob = global_rate / 3  # 3 mutations possible per base
            self.tv_prob = global_rate / 3
        if transition_rate is not None:
            self.tt_prob = transition_rate
        if transversion_rate is not None:
            self.tv_prob = transversion_rate / 2  # 2 transversions per base

        # dim=0 rows are target, dim=1 cols are source, 3 mutations per nucleotide
        # permutation is row index + 1, e.g. row 0 is permute by 1, row 1 permute by 2, etc.
        mutation_pmf = torch.full([3, 4], self.tv_prob)
        self._set_transition_pmf_(mutation_pmf, self.tt_prob)

        if nucleotide_rates is not None:
            for src, v in nucleotide_rates.items():
                for tgt, prob in v.items():
                    matrix_idx = (BASE_TO_INDEX[tgt] - BASE_TO_INDEX[src]) % N_BASE
                    mutation_pmf[matrix_idx - 1, BASE_TO_INDEX[src]] = prob

        # convert to cumulative probabilities by summing along target axis
        self.mutation_cmf = torch.cumsum(mutation_pmf, dim=0)
        assert torch.all(self.mutation_cmf >= 0.) and torch.all(self.mutation_cmf <= 1.)

    @staticmethod
    def _set_transition_pmf_(mutation_pmf, transition_prob):
            A = BASE_TO_INDEX['A']
            G = BASE_TO_INDEX['G']
            C = BASE_TO_INDEX['C']
            T = BASE_TO_INDEX['T']
            # transitions affect A<->G and C<->T
            mutation_pmf[((G - A) % N_BASE) - 1, A] = transition_prob
            mutation_pmf[((A - G) % N_BASE) - 1, G] = transition_prob
            mutation_pmf[((T - C) % N_BASE) - 1, C] = transition_prob
            mutation_pmf[((C - T) % N_BASE) - 1, T] = transition_prob

    def gen_variants(self, sequence, key, coord, omit_positions=None, as_pyvcf_format=False):
        uniform = torch.rand_like(sequence, dtype=torch.float)  # probabilities to compare to
        prob_cutoffs = torch.stack([torch.gather(row, -1, sequence) for row in self.mutation_cmf], dim=0)
        if omit_positions is not None:
            prob_cutoffs[..., (omit_positions - coord)] = 0.  # zero probability of any mutation at omit_positions
        cumulative = torch.sum(uniform < prob_cutoffs, dim=0).to(torch.long)
        indexes = torch.nonzero(cumulative, as_tuple=False).squeeze()
        chrom = [key] * indexes.size(-1)
        pos = indexes + coord
        ref = sequence[indexes]
        # subtract cumulative from sequence because of uniform < row comparison
        alt = torch.remainder(ref - cumulative[indexes], N_BASE)
        if as_pyvcf_format:
            return [self.PyVCFRecord(c, p.item(), r.item(), a.item()) for c, p, r, a in zip(chrom, pos, ref, alt)]
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
            self.ALT = [INDEX_TO_BASE[alt]]


# TODO indels
