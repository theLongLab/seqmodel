import sys
sys.path.append('./src')
import torch

from seqmodel import BASE_TO_INDEX, N_BASE


class MutationModel():

    """
    Sequence should contain bases in indexed form (i.e. tensor of dtype=torch.long).
        return: tuple of chrom, pos, ref, alt columns (arrays)
    """
    def gen_variants(self, sequence, key, coord):
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
                nucleotide_rates=None,
                ):
        tt_prob, tv_prob = 0, 0
        if global_rate is not None:
            tt_prob = global_rate / 3  # 3 mutations possible per base
            tv_prob = global_rate / 3
        if transition_rate is not None:
            tt_prob = transition_rate
        if transversion_rate is not None:
            tv_prob = transversion_rate / 2  # 2 transversions per base

        # dim=0 rows are target, dim=1 cols are source, 3 mutations per nucleotide
        # permutation is row index + 1, e.g. row 0 is permute by 1, row 1 permute by 2, etc.
        mutation_pmf = torch.full([3, 4], tv_prob)
        self._set_transition_pmf_(mutation_pmf, tt_prob)

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

    def gen_variants(self, sequence, key, coord):
        uniform = torch.rand_like(sequence, dtype=torch.float)  # probabilities to compare to
        prob_cutoffs = [torch.gather(row, 0, sequence) for row in self.mutation_cmf]
        cumulative = torch.stack([uniform < row for row in prob_cutoffs], dim=0)
        inv_mutations = torch.sum(cumulative.to(torch.long), dim=0)  # subtract from sequence because of <
        indexes = torch.nonzero(inv_mutations, as_tuple=False).squeeze()
        chrom = key
        pos = indexes + coord
        ref = sequence[indexes]
        alt = torch.remainder(ref - inv_mutations[indexes], N_BASE)
        return chrom, pos, ref, alt


# TODO indels
