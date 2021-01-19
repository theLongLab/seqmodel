import sys
sys.path.append('./src')
import os.path
from argparse import ArgumentParser
import torch
import torch.nn as nn
import pytorch_lightning as pl

from exp.seqbert import TOKENS_BP, TOKENS_BP_IDX
from exp.seqbert.model import VariantDecoder, SeqBERTLightningModule, \
                            CheckpointEveryNSteps, PrintGradients, main
from seqmodel.functional.mask import mask_select
from seqmodel.functional.log import summarize, binary_correct
from seqmodel.seqdata.mapseq import RandomRepeatSequence
from seqmodel.seqdata.iterseq import StridedSequence, FastaVariantFile, bed_from_file
from seqmodel.seqdata.variant import gen_variants, calc_mutation_rates, apply_variants, \
                            FixedRateSubstitutionModel
from seqmodel.functional import bioseq_to_index


def cls_test_transform(ref_seq, metadata):
    chrom, pos = metadata
    ref_seq = bioseq_to_index(ref_seq)
    seq_len = ref_seq.size(-1)

    # generate fake variants
    model = FixedRateSubstitutionModel(global_rate=0.2)
    fake_variants = gen_variants(model, ref_seq, chrom, pos,
                    as_pyvcf_format=True, min_variants=1)
    var_seq = apply_variants(ref_seq, chrom, pos, fake_variants)
    var_pos = (var_seq != ref_seq)

    labels = torch.full_like(ref_seq, TOKENS_BP_IDX['f'])
    labels[var_pos] = TOKENS_BP_IDX['t']
    return ref_seq, var_seq, labels, (chrom, pos, 'random')

def variant_transform(seqs, metadata):
    ref_seq, var_seq, var_records, sample = seqs
    chrom, pos = metadata
    ref_seq = bioseq_to_index(ref_seq)
    var_seq = bioseq_to_index(var_seq)
    seq_len = ref_seq.size(-1)
    real_var_pos = (var_seq != ref_seq)

    # generate fake variants with empirical mutation rates in this region
    ts_rate, tv_rate = calc_mutation_rates(ref_seq, var_seq)
    # do not generate in known variant positions (convert to zero indexing)
    omit_indexes = [x.POS - pos - 1 for x in var_records]
    model = FixedRateSubstitutionModel(transition_rate=ts_rate, transversion_rate=tv_rate)
    fake_variants = gen_variants(model, ref_seq, chrom, pos,
                omit_indexes=omit_indexes, as_pyvcf_format=True, min_variants=1)
    combined_var_seq = apply_variants(var_seq, chrom, pos, fake_variants)
    fake_var_pos = (combined_var_seq != var_seq)

    labels = torch.full_like(ref_seq,  TOKENS_BP_IDX['n'])
    labels[fake_var_pos] = TOKENS_BP_IDX['f']
    labels[real_var_pos] = TOKENS_BP_IDX['t']
    return ref_seq, combined_var_seq, labels, (chrom, pos, sample)


class PretrainCADD(SeqBERTLightningModule):

    def __init__(self, **hparams):
        super().__init__(VariantDecoder, **hparams)
        self.loss_fn = nn.BCEWithLogitsLoss()

        # get sequence of length 2*seq_len from dataloader
        # self.load_seq_len = self.hparams.seq_len_source_multiplier * self.hparams.seq_len
        # min_crop = int(self.hparams.seq_len * self.hparams.crop_factor)
        # max_crop = self.hparams.seq_len - min_crop
        # offset_min = 1 + min_crop
        # offset_max = self.hparams.seq_len - min_crop
        self.sample_freq = int(self.hparams.seq_len * self.hparams.seq_len_sample_freq)

    def get_dataloader(self, intervals_file, is_sequential):
        if self.hparams.DEBUG_use_random_data:
            data = RandomRepeatSequence(self.hparams.seq_len, n_batch=10000,
                                n_repeats=self.hparams.DEBUG_random_n_repeats,
                                repeat_len=self.hparams.DEBUG_random_repeat_len,
                                transform=cls_test_transform)
            return torch.utils.data.DataLoader(data, batch_size=self.hparams.batch_size,
                                shuffle=True, num_workers=self.hparams.num_workers)
        else:
            file_source = FastaVariantFile(self.hparams.seq_file, self.hparams.vcf_file)
            if intervals_file is not None:
                intervals = bed_from_file(intervals_file)
            dataset = StridedSequence(file_source, self.hparams.seq_len,
                                include_intervals=intervals,
                                transform=variant_transform,
                                sequential=is_sequential,
                                sample_freq=self.sample_freq)
            return dataset.get_data_loader(self.hparams.batch_size, self.hparams.num_workers)

    def train_dataloader(self):
        return self.get_dataloader(self.hparams.train_intervals, False)

    def val_dataloader(self):
        return self.get_dataloader(self.hparams.valid_intervals, True)

    def forward(self, batch):
        source, target, labels, _ = batch
        predicted, latent, embedded = self.model.forward(source, target)
        mask = (labels != TOKENS_BP_IDX['n'])
        classification_labels = (labels == TOKENS_BP_IDX['t']).float()
        predicted_variants = mask_select(predicted.squeeze(), mask)
        labelled_variants = mask_select(classification_labels, mask)
        if predicted_variants.size(-1) > 0:
            loss = self.loss_fn(predicted_variants, labelled_variants)
        else:
            loss = torch.tensor([0.])
        return loss, predicted, latent, source, target, embedded, labels, \
                mask, predicted_variants, labelled_variants

    def training_step(self, batch, batch_idx):
        loss, predicted, latent, source, target, embedded, labels, \
                mask, predicted_variants, labelled_variants = self.forward(batch)
        # if batch_idx > 1000 and loss > 2.2:
        #     print(loss, self.prev_loss)
        #     self.print_progress(loss, predicted, latent, source, target, mask, embedded, seqname, coord, include_grad=True)
        #     raise Exception("loss delta exceeded")
        self.prev_loss = loss.item()
        if batch_idx % self.hparams.print_progress_freq == 0:
            self.print_progress(loss, predicted, latent, source, target, embedded,
                labels, mask, predicted_variants, labelled_variants)
        return {'loss': loss, #'seqname': seqname[-1], 'coord': coord[-1],
                'log': {'train_loss': loss,} #'seqname': seqname[-1], 'coord': coord[-1],},
                }

    def print_progress(self, loss, predicted, latent, source, target, embedded, labels,
                        mask, predicted_variants, labelled_variants, include_grad=False):
        str_train_sample = summarize(
            labels,
            source,
            target,
            predicted.permute(1, 0, 2),
            index_symbols=TOKENS_BP)  # extra symbols represent masking
        classification_labels = (labels == TOKENS_BP_IDX['t']).float()
        n_true = torch.sum(classification_labels).item()
        n_false = predicted_variants.size(-1) - n_true
        n_vars = n_true + n_false + 0.0001  # add small number to avoid div by 0
        threshold = 0.5
        tp, tn, fp, fn = binary_correct(predicted_variants, labelled_variants, threshold)
        print('n:{} t:{} f:{}'.format(int(predicted_variants.size(-1)), int(n_true), int(n_false)),
            'tp:{} tn:{} fp:{} fn:{}'.format(tp / n_vars, tn / n_vars, fp / n_vars, fn / n_vars),
            str_train_sample, sep='\n')

    def validation_step(self, batch, batch_idx):
        loss, predicted, latent, source, target, embedded, labels, \
                mask, predicted_variants, labelled_variants = self.forward(batch)
        if batch_idx % self.hparams.print_progress_freq == 0:
            self.print_progress(loss, predicted, latent, source, target, embedded,
                labels, mask, predicted_variants, labelled_variants)
        return {'loss': loss,
                'log': {
                    'val_loss': loss,
                    # 'correct': acc_numbers,
                    # 'train_sample': str_train_sample,
                }}

    # def validation_epoch_end(self, val_step_outputs):
    #     result = pl.EvalResult(checkpoint_on=loss)
    #     result.log('val_loss', loss)

    @staticmethod
    def add_model_specific_args(parent_parser):
        super_parser = SeqBERTLightningModule.add_model_specific_args(parent_parser)
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[super_parser])
        # data params
        parser.add_argument('--seq_file', default='data/ref_genome/p12/assembled_chr/GRCh38_p12_assembled_chr.fa', type=str)
        parser.add_argument('--vcf_file', default='data/vcf/ALL.chr22.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz', type=str)
        parser.add_argument('--train_intervals', default=None, type=str)
        parser.add_argument('--valid_intervals', default=None, type=str)
        # parser.add_argument('--seq_len_source_multiplier', default=2., type=float)  # how much length to add when loading
        # parser.add_argument('--crop_factor', default=0.2, type=float)  # how much of source sequence to keep when cropping
        parser.add_argument('--seq_len_sample_freq', default=0.5, type=float)  # gives sample_freq in StridedSequence
        parser.add_argument('--DEBUG_use_random_data', default=False, type=bool)
        parser.add_argument('--DEBUG_random_repeat_len', default=1, type=int)
        parser.add_argument('--DEBUG_random_n_repeats', default=500, type=int)
        return parser


if __name__ == '__main__':
    main(PretrainCADD, is_classifier=False)
