import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from Bio import SeqIO
from utils import seq_to_ohe

class LuciferaseDataset(Dataset):

    def __init__(self, fasta_path, seq_len=360, charset=21):

        self.records = SeqIO.parse(fasta_path, 'fasta')

        self.ohe_records = np.stack([self.encode_seq(record) for record in self.records])
        self.ohe_records = self.ohe_records.reshape(-1, charset, seq_len)

    def encode_seq(self, record):

        seq = str(record.seq)
        ohe = seq_to_ohe(seq)

        return ohe

    def __len__(self):

        return len(self.ohe_records)

    def __getitem__(self, idx):

        x = self.ohe_records[idx, ...]

        return x


if __name__ == '__main__':

    dataset = LuciferaseDataset('./data/luxafilt_llmsa_train.fa')
    print(dataset[0])
    print(dataset[0].shape)
