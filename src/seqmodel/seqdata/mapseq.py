import sys
sys.path.append('./src')
import random
import torch
import torch.nn.functional as F
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq, MutableSeq
from Bio.Alphabet import generic_dna
from seqmodel.functional.transform import bioseq_to_index, INDEX_TO_BASE


def random_bioseq(length, alphabet=INDEX_TO_BASE):
    return Seq(''.join([random.choice(alphabet) for _ in range(length)]))


# map style dataset holding entire sequence in memory
class MapSequence(torch.utils.data.Dataset):

    def __init__(self, bioseq, seq_len, stride=1, overlap=None, remove_gaps=False):
        self.bioseq = bioseq.upper()
        if remove_gaps:
            self.bioseq = self.bioseq.ungap('N')
        self.augment_state = 0
        if overlap is None:
            self.stride = stride
        else:
            self.stride = seq_len - overlap
        self.seq_len = seq_len

    @classmethod
    def from_file(cls, fasta_filename, seq_len, stride=1, overlap=None, remove_gaps=False):
        bioseq = SeqIO.read(fasta_filename, "fasta").seq
        return cls(bioseq, seq_len, stride=1, overlap=overlap, remove_gaps=remove_gaps)

    @property
    def seq_len(self):
        return self._seq_len

    @seq_len.setter
    def seq_len(self, value):
        self._seq_len = value
        self._total_len = int((len(self.bioseq) - self._seq_len + self.stride) / self.stride)

    def __len__(self):
        return self._total_len

    def __getitem__(self, index):
        subseq = self.bioseq[index * self.stride : index * self.stride + self.seq_len]
        return bioseq_to_index(subseq)


class RandomRepeatSequence(torch.utils.data.Dataset):

    def __init__(self, seq_len, n_batch, n_repeats, repeat_len=1):
        self.seq_len = seq_len
        self.n_batch = n_batch
        self.n_repeats = n_repeats
        self.repeat_len = repeat_len

        empty_seq = Seq('')
        self.seq = MutableSeq('')
        random.seed(0)
        while len(self.seq) < self.seq_len * self.n_batch:
            arr = [random_bioseq(self.repeat_len)] * self.n_repeats
            self.seq.extend(empty_seq.join(arr))

    def __len__(self):
        return self.n_batch

    def __getitem__(self, index):
        seq = bioseq_to_index(self.seq[index * self.seq_len: (index + 1) * self.seq_len])
        return seq, ('Random', index)


# quickly create a batch for testing purposes
def create_test_batch(batch_size, seq_len):
    dataset = RandomRepeatSequence(seq_len, batch_size, 2, repeat_len=2)
    return torch.stack([dataset[i][0] for i in range(batch_size)], dim=0)


# class LabelledSequence(torch.utils.data.Dataset):

#     def __init__(self, filename, input_seq_len):
#         data = torch.load(filename)
#         self.labels = torch.tensor(data['labels'][:, (891, 914)])
#         self.one_hot = torch.tensor(data['x'][:, :, 500-int(input_seq_len/2):500+int(input_seq_len/2)])

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, index):
#         return self.one_hot[index], self.labels[index]
