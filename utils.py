import numpy as np
import torch

aa_list = 'ACDEFGHIKLMNPQRSTVWY-'

def seq_to_ohe(seq):

    arr = np.zeros((len(seq), len(aa_list)), dtype=int)
    for i, c in enumerate(seq):
        arr[i, aa_list.index(c)] = 1

    return arr

def ohe_to_seq(arr):

    indices = arr.argmax(axis=1)
    seq = ''.join([aa_list[idx] for idx in indices])

    return seq


if __name__ == '__main__':

    print(len(aa_list))
    # seq = 'ACERTPLEW'
    # arr = seq_to_ohe(seq)
    # print(arr)

    # seq = ohe_to_seq(arr)
    # print(seq)
