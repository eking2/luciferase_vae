import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import logging

from dataset import LuciferaseDataset
from models import fc_encoder, fc_decoder, MSA_VAE
from utils import ohe_to_seq, seq_to_ohe, aa_list, save_model, setup_logger, luxA
from collections import defaultdict


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

    epoch_loss = 0.
    epoch_acc = 0.

    model.eval()

    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for batch_idx, x in loop:
        x = x.to(DEVICE).float()

        x_rec, mu, logvar = model(x)
        loss = model.vae_loss(x, x_rec, mu, logvar)

        # acc over each char
        pred_labels = x_rec.argmax(dim=1)
        true_labels = x.argmax(dim=1)

        correct = (pred_labels == true_labels).sum().float()
        acc = correct / true_labels.numel()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)


def check_prog(model):

    model.eval()

    fixed_ohe = torch.tensor(seq_to_ohe(luxA)).reshape(1, 21, 360).float().to(DEVICE)
    x_rec, mu, logvar = model(fixed_ohe)
    out = x_rec.detach().cpu().numpy()
    out_seq = ohe_to_seq(out.reshape(360, 21))

    acc = sum([(i == j) for i, j in zip(luxA, out_seq)])  / len(luxA)

    return out_seq, acc


def train_loop(model, train_loader, valid_loader, optimizer, epochs):

    history = defaultdict(list)

    for epoch in range(epochs):

        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader)
        valid_loss, valid_acc = eval(model, valid_loader)
        fixed_input, acc = check_prog(model)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)

        logging.info(f'Epoch: {epoch+1}')
        logging.info(f'Train Loss: {train_loss:.2f}  | Train acc: {train_acc:.2f}')
        logging.info(f'Valid Loss: {valid_loss:.2f}  | Valid acc: {valid_acc:.2f}')
        logging.info(f'Fixed input: {fixed_input}  | Acc: {acc:.2f}')

    save_model(model, optimizer, epoch+1)


if __name__ == '__main__':

    setup_logger('luxa') 

    logging.info(f'batch size: {BATCH_SIZE}')
    logging.info(f'epochs: {EPOCHS}')
    logging.info(f'train fasta: {TRAIN_FASTA}')
    logging.info(f'valid fasta: {VALID_FASTA}')
    logging.info(f'lr: {LR}')
    logging.info(f'encoder kwargs: {ENCODER_KWARGS}')
    logging.info(f'decoder kwargs: {DECODER_KWARGS}')
    logging.info(f'device: {DEVICE}')

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
    train_loop(model, train_loader, valid_loader, optimizer, EPOCHS)



