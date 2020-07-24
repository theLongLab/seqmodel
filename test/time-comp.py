import timeit
import torch
import torch.nn.functional as F


if __name__ == '__main__':
    do_print_memory_use = False

    def print_memory_use():
        if do_print_memory_use:
            print_memory_use()

    dev = torch.device('cuda')
    n_tries = 1000
    if do_print_memory_use:
        n_tries = 1

    for seq_len in [100, 1000, 10000]:
        def gen1():
            x = torch.tensor([0]*int(0.4 * seq_len) + [1]*int(0.3 * seq_len) + [2]* int(0.2 * seq_len)+ [3] * int(0.1 * seq_len), device=dev)
            print_memory_use()
            return x

        def gen2():
            x = torch.cat((torch.zeros(int(0.4 * seq_len), device=dev),torch.ones(int(0.3 * seq_len), device=dev),torch.ones(int(0.2 * seq_len), device=dev) * 2, torch.ones(int(0.1 * seq_len), device=dev) * 3))
            print_memory_use()
            return x

        for fn in [gen1, gen2]:
            print(timeit.timeit(lambda: fn(), number=n_tries), seq_len, fn)

        def mask1():
            # mask = gen1()
            mask = gen2()
            indexes = torch.randperm(seq_len, device=dev)
            mask = mask[indexes]
            print_memory_use()
            return mask

        def mask2_gen():
            prob = torch.rand(seq_len, device=dev)
            mask = torch.zeros(seq_len, device=dev)
            print_memory_use()
            return mask, prob

        def fill1():
            mask, prob = mask2_gen()
            mask = mask.masked_fill(prob > 0.4, 1)
            mask = mask.masked_fill(prob > 0.7, 2)
            mask = mask.masked_fill(prob > 0.9, 3)
            x = torch.sum(mask)
            print_memory_use()
            return x

        def fill2():
            mask, prob = mask2_gen()
            mask[prob > 0.4] += 1
            mask[prob > 0.7] += 1
            mask[prob > 0.9] += 1
            x = torch.sum(mask)
            print_memory_use()
            return x

        def fill_few():
            mask, prob = mask2_gen()
            mask = mask.masked_fill(prob > 0.05, 1)
            mask = mask.masked_fill(prob > 0.05, 2)
            mask = mask.masked_fill(prob > 0.05, 3)
            x = torch.sum(mask)
            print_memory_use()
            return x

        def fill_lots():
            mask, prob = mask2_gen()
            mask = mask.masked_fill(prob > 0.95, 1)
            mask = mask.masked_fill(prob > 0.95, 2)
            mask = mask.masked_fill(prob > 0.95, 3)
            x = torch.sum(mask)
            print_memory_use()
            return x

        def fill_sum():
            mask, prob = mask2_gen()
            mask = (prob > 0.4).type(torch.int8) + (prob > 0.7).type(torch.int8) + (prob > 0.9).type(torch.int8)
            x = torch.sum(mask)
            print_memory_use()
            return x

        for fn in [mask1, fill1, fill2, fill_sum]:
            print(timeit.timeit(lambda: fn(), number=n_tries), seq_len, fn)

        for batches in [1, 10, 100]:

            def one_hot(index_sequence, indexes=range(4), dim=1):
                with torch.no_grad():
                    x = torch.stack([(index_sequence == i).float() for i in indexes], dim=dim)
                    print_memory_use()
                    return x

            def one_hot1(index_sequence):
                with torch.no_grad():
                    output = torch.zeros(index_sequence.size(0), 4,
                            index_sequence.size(1), device=index_sequence.device)
                    for i in range(4):
                        output[:, i,:] = output[:, i,:].masked_fill(index_sequence == i, float(1.0))
                    print_memory_use()
                return output

            def one_hot_ref(x):
                x = F.one_hot(x, num_classes=4).permute(0, 2, 1).type(torch.float32)
                print_memory_use()
                return x

            def one_hot_test(fn, batches, seq_len):
                batch = torch.randint(4, (batches, seq_len), device=dev)
                return fn(batch)

            for fn in [one_hot, one_hot1, one_hot_ref]:
                        print(timeit.timeit(lambda: one_hot_test(fn, batches, seq_len), number=n_tries),
                                batches, seq_len, fn)
