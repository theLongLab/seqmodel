import sys
sys.path.append('./src')

from seqmodel import INDEX_TO_BASE

TOKENS_BP = INDEX_TO_BASE + [  # AGCT 0 1 2 3
    'n',  # 4 unknown base
    'm',  # 5 masked base
    '~',  # 6 classification token (always at start)
    'f',  # 7 output token at classification token position, indicates pretext task false
    't',  # 8 output token indicating pretext task is true
    '/',  # 9 separator token (between two input sequences)
    ]

TOKENS_BP_IDX = {k: v for v, k in enumerate(TOKENS_BP)}  # dict to look up above
