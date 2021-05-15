import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class fc_encoder(nn.Module):

    def __init__(self, seq_len, latent_dim, alphabet_size=21, encoder_hidden=[250, 250, 250],
                 encoder_dropout=[0.7, 0, 0], activation='ReLU', n_conditions=0):

        super().__init__()

        layers = []

        # conditional net has concatenated one hot input
        input_dim = seq_len * alphabet_size
        if n_conditions > 0:
            input_dim += n_conditions

        # activation func from str
        act = getattr(nn, activation)

        for i, (n_hid, drop) in enumerate(zip(encoder_hidden, encoder_dropout)):

            if i == 0:
                layer = nn.Linear(input_dim, n_hid)

            else:
                layer = nn.Linear(encoder_hidden[i-1], n_hid)

            layers.append(layer)
            layers.append(act())

            if drop > 0:
                dropout = nn.Dropout(drop)
                layers.append(dropout)

        self.net = nn.Sequential(*layers)
        self.z_mean = nn.Linear(encoder_hidden[-1], latent_dim)
        self.z_var = nn.Linear(encoder_hidden[-1], latent_dim)


    def forward(self, x):

        x = self.net(x)

        z_mean = self.z_mean(x)
        z_var = self.z_var(x)

        return z_mean, z_var


class fc_decoder(nn.Module):

    def __init__(self, seq_len, latent_dim, alphabet_size=21, decoder_hidden=[250], decoder_dropout=[0],
                 activation='ReLU', n_conditions=0):

        super().__init__()

        layers = []

        input_dim = latent_dim
        if n_conditions > 0:
            input_dim += n_conditions

        for i, (n_hid, drop) in enumerate(zip(decoder_hidden, decoder_dropout)):

            if i == 0:
                layer = nn.Linear(input_dim, n_hid)

            else:
                layer = nn.Linear(decoder_hidden[i-1], n_hid)

            layers.append(layer)
            layers.append(act())

            if drop > 0:
                dropout = nn.Dropout(drop)
                layers.append(dropout)

        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(decoder_hidden[-1], seq_len * alphabet_size)


    def forward(self, x):

        x = self.net(x)
        x = self.fc(x)
        x = x.view(x.shape[0], self.alphabet_size, self.seq_len)

        return x


class cnn_encoder(nn.Module):

    pass


class MSA_VAE(nn.Module):

    def __init__(self):
        pass


class AR_VAE(nn.Module):

    pass



if __name__ == '__main__':

    mod = fc_encoder(360, 10)
    print(mod)
