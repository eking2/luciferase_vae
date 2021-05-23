import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import LuciferaseDataset
from models import fc_encoder, fc_decoder, MSA_VAE
from utils import ohe_to_seq, aa_list


BATCH_SIZE = 32
EPOCHS =  14
TRAIN_FASTA = './data/luxafilt_llmsa_train.fa'
VALID_FASTA = './data/luxafilt_llmsa_val.fa'

ENCODER_KWARGS = {'latent_dim' : 10,
        'seq_len' : 360,
        'encoder_hidden' : [256, 256],
        'encoder_dropout' : [0, 0],
        }

DECODER_KWARGS = {'latent_dim' : 10,
        'seq_len' : 360,
        'decoder_hidden' : [256, 256],
        'decoder_dropout' : [0, 0]}

LR = 0.001

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, optimizer, loader):

    epoch_loss = 0.
    epoch_acc = 0.

    model.train()

    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for batch_idx, x in loop:
        x = x.to(DEVICE).float()

        x_rec, mu, logvar = model(x)
        loss = model.vae_loss(x, x_rec, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # acc over each char
        pred_labels = x_rec.argmax(dim=1)
        true_labels = x.argmax(dim=1)

        correct = (pred_labels == true_labels).sum().float()
        acc = correct / true_labels.numel()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)


def eval(model, loader):

    pass


def train_loop(model, train_loader, valid_loader, optimizer, epochs):

    pass



if __name__ == '__main__':

    # setup data
    train_dataset = LuciferaseDataset(TRAIN_FASTA)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    valid_dataset = LuciferaseDataset(VALID_FASTA)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # setup msa_vae model
    encoder = fc_encoder(**ENCODER_KWARGS)
    decoder = fc_decoder(**DECODER_KWARGS)

    model = MSA_VAE(encoder, decoder)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # train
    for epoch in range(EPOCHS):
        loss, acc = train_one_epoch(model, optimizer, train_loader)
        print(epoch, loss, acc)



