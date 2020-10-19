from math import ceil
import torch
import torch.nn.functional as F
from seqmodel.functional.transform import one_hot_to_index, softmax_to_index, N_BASE, INDEX_TO_BASE


SYMBOL_INT = [' '] + [str(x) for x in range(1, 10)] + ['0']
SYMBOL_MAGNITUDES = [' ', '.', '-', '~', ':', '<', '*', '^', '#', '$', '@']
SYMBOL_BOOL = ['!', '.']
EPSILON = 1e-9


def softmax_dim(dim):
    return lambda x: F.softmax(x, dim=dim)

def correct(predicted, target, threshold_score=float('-inf')):
    with torch.no_grad():
        predicted_index = softmax_to_index(predicted, threshold_score=threshold_score)
        return (target - predicted_index) == 0

def n_correct(predicted, target, threshold_score=float('-inf')):
    with torch.no_grad():
        indexes = correct(predicted, target, threshold_score=threshold_score)
        return torch.sum(indexes).item()

def accuracy(predicted, target, threshold_score=float('-inf')):
    with torch.no_grad():
        return n_correct(predicted, target,
            threshold_score=threshold_score) / target.nelement()

def prediction_histograms(predicted, target, n_class=N_BASE, n_bins=10,
                        normalize_fn=softmax_dim(1)):
    with torch.no_grad():
        if len(target.shape) == 1:  # add dummy batch dimension
            predicted = predicted.unsqueeze(0)
            target = target.unsqueeze(0)
        values, indexes = torch.max(normalize_fn(predicted), dim=1)
        correct_mask = (target - indexes) == 0
        incorrect_mask = torch.logical_not(correct_mask)
        hist_min = 1. / n_class
        hist_max = 1.
        # dimensions are (incorrect/correct, class, confidence bins)
        histograms = []
        for mask in [incorrect_mask, correct_mask]:
            sub_hist = []
            for class_index in range(n_class):
                masked_values = torch.masked_select(values, (target == class_index) * mask)
                hist = torch.histc(masked_values, bins=n_bins, min=hist_min, max=hist_max)
                sub_hist.append(hist)
            histograms.append(torch.stack(sub_hist, dim=0))
        return torch.stack(histograms, dim=0)

def normalize_histogram(histogram, weights=None):
    if weights is None:
        weights = histogram
    return (histogram.permute(2, 0, 1) / (torch.sum(weights, dim=2) + EPSILON)).permute(1, 2, 0)

def accuracy_per_class(histogram, threshold_prob=0.5):
    n_class = histogram.shape[1]
    n_bins = histogram.shape[2]
    min_prob = 1. / n_class
    cutoff_bin = int(ceil(threshold_prob / ((1. - min_prob) / n_bins)))
    sums = torch.sum(histogram[:, :, cutoff_bin:], dim=2, keepdim=True)
    return torch.squeeze(normalize_histogram(sums, weights=histogram)[1,:,:])

def broadcastable_dims(*tensors):
    n_dims = [len(t.shape) for t in tensors]
    dims = tensors[n_dims.index(max(n_dims))].shape  # tensor with most dims
    for t in tensors:  # check all tensor trailing dimensions are identical
        for d in dims:
            if t.shape != dims[-1 * len(t.shape):len(dims)]:
                return None, None
    return n_dims, dims

# tensors must be broadcastable
def excerpt(*tensors, max_sizes=[], random_pos=False):
    with torch.no_grad():
        tensors = list(tensors)
        n_dims, dims = broadcastable_dims(*tensors)
        out_dims = len(max_sizes)
        assert not (n_dims is None)
        if out_dims < len(dims):  # pad max_sizes with 1s
            max_sizes = [1] * (len(dims) - out_dims) + max_sizes
        else:
            max_sizes = max_sizes[-len(dims):]
        for i, (dim, max_size) in enumerate(zip(dims, max_sizes)):
            offset = 0
            length = min(dim, max_size)
            max_offset = max(0, dim - max_size)
            if random_pos and max_offset > 0:
                offset = torch.randint(max_offset, [1]).item()
            for j, (n_dim, t) in enumerate(zip(n_dims, tensors)):
                dim_diff = len(dims) - n_dim
                if dim_diff <= i:
                    tensors[j] = torch.narrow(t, i - dim_diff, offset, length)
        # make every tensor have same number of dims
        for i, (n_dim, t) in enumerate(zip(n_dims, tensors)):
            dim_diff = out_dims - n_dim
            if dim_diff > 0:
                tensors[i] = t.view([1] * dim_diff + list(t.shape))
            else:
                tensors[i] = t.view(t.shape[-1 * out_dims:])
        return tensors

# tensors must be broadcastable
def summarize(*tensors, random_pos=False, max_lines_per_tensor=5, max_len=90, index_symbols=INDEX_TO_BASE,
                row_labels=None, col_labels=None, normalize_fn=softmax_dim(0)):
    with torch.no_grad():
        _, dims = broadcastable_dims(*tensors)
        assert not (dims is None)

        if row_labels is None:  # get printing bounds
            label_len = 1
        else:
            label_len = max([max([len(l) for l in labels]) for labels in row_labels])
        max_len = max_len - label_len - 1
        max_col = max(max_len // (dims[-1] + 1), min(2, dims[-2]))  # +1 for space between batches
        tensors = excerpt(*tensors,
                    max_sizes=[max_lines_per_tensor, max_col, max_len // max_col - 1],
                    random_pos=random_pos)

        # if no labels are given, label 0th dim as either ' ' or as index_symbols
        if row_labels is None:
            row_labels = []
            for t in tensors:
                if t.shape[0] == len(index_symbols):
                    row_labels.append(index_symbols)
                else:
                    row_labels.append([' '] * t.shape[0])

        # the order of printing is (lines, columns, rows in each column), i.e.:
        # 000, 001 | 010, 011
        # 100, 101 | 110, 111
        output_str = []
        if not (col_labels is None):
            assert len(col_labels) == tensors[0].shape[1]
            col_len = tensors[0].shape[2]
            col_labels = [label[:col_len] + ''.join(['_'] * (col_len - len(label)))
                        for label in col_labels]
            output_str += str_cols(col_labels, label_len, label='', symbols=None, v_break=' ')
        for labels, t in zip(row_labels, tensors):
            # infer str representation from tensor type
            if t.dtype == torch.bool:  # boolean tensor
                symbols = SYMBOL_BOOL
            elif isint(t):  # assume ints are index sequence
                symbols = index_symbols
            elif isfloat(t):  # normalize floats between 0. and 1.
                if not (normalize_fn is None):
                    t = normalize_fn(t)
                t = (t * 10).type(torch.long)
                symbols = SYMBOL_MAGNITUDES
            else:
                symbols = None
            for label, line in zip(labels, t):  # print labels and each column on the same line
                output_str += str_cols(line, label_len, label=label, symbols=symbols)
        return ''.join(output_str[:-1])  # remove last \n

def str_cols(col_list, label_len, label='', symbols=SYMBOL_INT, v_break='|'):
    output_str = [' '] * (label_len - len(label)) + [label] + \
                [v_break + str_summary(col, symbols=symbols) for col in col_list] + \
                [v_break] + ['\n']
    return output_str

# outputs a string corresponding to a flattened tensor
def str_summary(tensor, symbols=SYMBOL_INT, default_symbol='X'):
        if symbols is None:
            return str(tensor)
        tensor = tensor.flatten()
        symbol_dict = {i : symbols[i] for i in range(len(symbols))}
        return ''.join([symbol_dict.get(j.item(), default_symbol) for j in tensor])

def isint(tensor):
    return tensor.dtype == torch.uint8 or tensor.dtype == torch.int8 or \
            tensor.dtype == torch.int16 or tensor.dtype == torch.int32 or \
            tensor.dtype == torch.int64

def isfloat(tensor):
    return tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16 or \
            tensor.dtype == torch.float32 or tensor.dtype == torch.float64

def summarize_weights_and_grads(module_dict, include_grad=False, threshold_trigger=0.):
    output_str = ''
    do_output = False
    for name, module in module_dict.items():
        output_str += str(name) + str(module)
        for param in module.parameters():
            if torch.abs(torch.max(param)).item() > threshold_trigger \
                    or torch.abs(torch.min(param)).item() > threshold_trigger:
                do_output = True
            output_str += str(param.shape) + tensor_stats_str(param, include_grad=include_grad) + '\n'
    if do_output:
        return output_str + '\n'
    return ''

def get_stats(tensor):
    std, mean = torch.std_mean(tensor)
    return torch.min(tensor).item(), mean.item(), std.item(), torch.max(tensor).item()

def tensor_stats_str(*tensors, include_grad=False):
    if include_grad:
        strings = ['<{:0.2f} [{:0.2f}/{:0.2f}] {:0.2f}> ({:0.2f} {{{:0.2f}/{:0.2f}}} {:0.2f})'.format(
                *get_stats(t), *get_stats(t.grad)) for t in tensors]
    else:
        strings = ['<{:0.2f} [{:0.2f}/{:0.2f}] {:0.2f}>'.format(
                *get_stats(t)) for t in tensors]
    return ' || '.join(strings)
