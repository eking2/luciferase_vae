import numpy as np
import torch
import logging

aa_list = 'ACDEFGHIKLMNPQRSTVWY-'

luxA = 'MKFGNFLLTYQPPQFSQTEVMKRLVKLGRISEECGFDTVWLLEHHFTEFGLLGNPYVAAAYLLGATKKLNVGTAAIVLPTAHPVRQLEDVNLLDQMSKGRFRFGICRGLYNKDFRVFGTDMNNSRALAECWYGLIKNGMTEGYMEADNEHIKFHKVKVNPAAYSRGGAPVYVVAESASTTEWAAQFGLPMILSWIINTNEKKAQLELYNEVAQEYGHDIHNIDHCLSYITSVDHDSIKAKEICRKFLGHWYDSYVNATTIFDDSDQTRGYDFNKGQWRDFVLKGHKDTNRRIDYSYEINPVGTPQECIDIIQKDIDATGISNICCGFEANGTVDEIIASMKLFQSDVMPFLKEKQRSLLY' 

def seq_to_ohe(seq):

    arr = np.zeros((len(seq), len(aa_list)), dtype=int)
    for i, c in enumerate(seq):
        arr[i, aa_list.index(c)] = 1

    return arr

def ohe_to_seq(arr):

    indices = arr.argmax(axis=1)
    seq = ''.join([aa_list[idx] for idx in indices])

    return seq

def save_model(model, optimizer, epoch):

    torch.save({'epoch' : epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict()},
                f'checkpoints/luciferase_{epoch}.pt')

def setup_logger(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handlers = [logging.StreamHandler(),
                logging.FileHandler(f'logs/{log_file}.log', 'a')]

    # do not print millisec
    fmt = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s',
                            "%Y-%m-%d %H:%M:%S")

    for h in handlers:
        h.setFormatter(fmt)
        logger.addHandler(h)

    return logger


if __name__ == '__main__':

    print(len(aa_list))
    print(luxA)
    print(len(luxA))
    # seq = 'ACERTPLEW'
    # arr = seq_to_ohe(seq)
    # print(arr)

    # seq = ohe_to_seq(arr)
    # print(seq)
