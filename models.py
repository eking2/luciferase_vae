import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class fc_encoder(nn.Module):

    def __init__(self, seq_len, latent_dim, alphabet_size=21, encoder_hidden=[250, 250, 250],
                 encoder_dropout=[0.7, 0, 0], activation='ReLU', n_conditions=0):

        super().__init__()

        self.seq_len = seq_len
        self.alphabet_size = alphabet_size

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

        # flatten if (batch, charset, seq_len)
        if (x.shape[1] * x.shape[2]) == (self.seq_len * self.alphabet_size):
            x = x.view(x.shape[0], -1)

        x = self.net(x)

        mu = self.z_mean(x)
        logvar = self.z_var(x)

        return mu, logvar


class fc_decoder(nn.Module):

    def __init__(self, seq_len, latent_dim, alphabet_size=21, decoder_hidden=[250], decoder_dropout=[0],
                 activation='ReLU', n_conditions=0):

        super().__init__()

        self.seq_len = seq_len
        self.alphabet_size = alphabet_size

        layers = []

        act = getattr(nn, activation)

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

    def __init__(self, encoder, decoder):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps*std

    def forward(self, x):

        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decoder(z)

        return x_rec, mu, logvar

    def vae_loss(self, x, x_rec, mu, logvar):

        labels = x.argmax(dim=1)

        # x_rec is ohe
        recon_loss = F.cross_entropy(x_rec, labels, reduction='sum') / x.shape[0]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar)) / x.shape[0]

        return recon_loss + kl_loss


class AR_VAE(nn.Module):

    pass



if __name__ == '__main__':

    enc_input = torch.randn(2, 21, 360)
    encoder = fc_encoder(360, 10)
    mu, logvar = encoder(enc_input)
    print(mu.shape)
    print(logvar.shape)

    dec_input = torch.randn(2, 10)
    decoder = fc_decoder(360, 10)
    x_rec = decoder(dec_input)
    print(x_rec.shape)

    vae = MSA_VAE(encoder, decoder)
    x_rec, mu, logvar = vae(enc_input)
    print(x_rec.shape)

