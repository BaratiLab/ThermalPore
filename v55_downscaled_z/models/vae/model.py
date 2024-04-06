# Adapted from
# https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

import torch
import torch.nn as nn

from einops import rearrange
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            latent_dim,
            hidden_dims,
            verbose,
            name
        ):
        super(Model, self).__init__()
        self.verbose = verbose
        self.latent_dim = latent_dim

        modules = []

        # Build Encoder
        for hidden_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels = hidden_dim,
                        kernel_size = 3,
                        stride = 2,
                        padding  = 1
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = hidden_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride = 2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(
                hidden_dims[-1],
                out_channels=out_channels,
                kernel_size= 3,
                padding= 1
            ),
            # nn.Tanh()
            nn.Sigmoid()
        )

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        if self.verbose: print(f"self.encoder(input) {result.shape}")
        result = torch.flatten(result, start_dim=1)
        if self.verbose: print(f"torch.flatten(result, start_dim=1) {result.shape}")

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        input = rearrange(input, 'b c f h w -> b (c f) h w')
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        # return  [self.decode(z), input, mu, log_var]
        recons = self.decode(z)
        recons = rearrange(recons, 'b c h w -> b h w c')
        return  [recons, mu, log_var]
