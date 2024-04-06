# Adapted from
# https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

import torch
import torch.nn as nn
import numpy as np

from einops import rearrange
from torch.nn import functional as F

def cnn_out_dim(in_dim, padding_size, kernel_size, stride):
    return (in_dim + 2 * padding_size - kernel_size) // stride + 1

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
        self.hidden_dims = hidden_dims

        modules = []

        kernel_size = 3
        stride = 2
        padding = 1

        # Build Encoder
        for hidden_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        out_channels = hidden_dim,
                        kernel_size = 3,
                        stride = 2,
                        padding  = 1
                    ),
                    nn.BatchNorm3d(hidden_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = hidden_dim

        dims = [325, 190, 190]
        for index, dim in enumerate(dims):
            out_dim = dim 
            for _ in hidden_dims:
                out_dim = cnn_out_dim(out_dim, padding, kernel_size, stride)
            dims[index] = out_dim
        # print(dims)
        self.dims_latent = dims

        self.encoder = nn.Sequential(*modules)

        self.flattened_dim = hidden_dims[-1] * np.prod(self.dims_latent)

        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_var = nn.Linear(self.flattened_dim, latent_dim)
        # self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        # self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, self.flattened_dim)
        # self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        modules.append(
            nn.Sequential(
                nn.ConvTranspose3d(
                    self.flattened_dim//np.prod(self.dims_latent),
                    hidden_dims[1],
                    kernel_size=3,
                    stride = 2,
                    padding=1,
                    output_padding=(0, 1, 1)
                ),
                nn.BatchNorm3d(hidden_dims[1]),
                nn.LeakyReLU()
            )
        )

        for i in range(1, len(hidden_dims) - 4):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride = 2,
                        padding=1,
                        output_padding=(1, 1, 1)
                    ),
                    nn.BatchNorm3d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose3d(
                    hidden_dims[-4],
                    hidden_dims[-3],
                    kernel_size=3,
                    stride = 2,
                    padding=1,
                    output_padding=(0, 1, 1)
                ),
                nn.BatchNorm3d(hidden_dims[-3]),
                nn.LeakyReLU()
            )
        )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose3d(
                    hidden_dims[-3],
                    hidden_dims[-2],
                    kernel_size=3,
                    stride = 2,
                    padding=1,
                    output_padding=(1, 1, 1)
                ),
                nn.BatchNorm3d(hidden_dims[-2]),
                nn.LeakyReLU()
            )
        )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose3d(
                    hidden_dims[-2],
                    hidden_dims[-1],
                    kernel_size=3,
                    stride = 2,
                    padding=1,
                    output_padding=(0, 0, 0)
                ),
                nn.BatchNorm3d(hidden_dims[-1]),
                nn.LeakyReLU()
            )
        )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv3d(
                hidden_dims[-1],
                out_channels=out_channels,
                kernel_size=3,
                # stride =2,
                stride=1,
                padding=1,
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
        if self.verbose: print(f"self.fc_mu(result) {mu.shape}")
        log_var = self.fc_var(result)
        if self.verbose: print(f"self.fc_var(result) {log_var.shape}")

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        if self.verbose: print(f"self.decoder_input(z) {result.shape}")
        # result = result.view(-1, 512, 2, 2)
        # result = result.view(-1, 9216, 2, 2)
        result_shape = [-1, self.flattened_dim//np.prod(self.dims_latent), *self.dims_latent]
        result = result.view(*result_shape)
        if self.verbose: print(f"result.view(*result_shape) {result.shape}")
        result = self.decoder(result)
        if self.verbose: print(f"self.decoder(result) {result.shape}")
        result = self.final_layer(result)
        if self.verbose: print(f"self.final_layer(result) {result.shape}")
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
        if self.verbose: print(f"torch.exp(0.5 * logvar) {std.shape}")
        eps = torch.randn_like(std)
        if self.verbose: print(f"torch.randn_like(std) {eps.shape}")
        return eps * std + mu

    def forward(self, input):
        if self.verbose: print(f"Input: {input.shape}")
        # input = rearrange(input, 'b c f h w -> b (c f) h w')
        input = rearrange(input, 'b f h w -> b 1 f h w')
        if self.verbose: print(f"rearrange(input, 'b f h w -> b 1 f h w'): {input.shape}")
        mu, log_var = self.encode(input)
        if self.verbose: print(f"self.encode(input) {mu.shape} {log_var.shape}")
        z = self.reparameterize(mu, log_var)
        if self.verbose: print(f"self.reparameterize(mu, log_var) {z.shape}")
        # return  [self.decode(z), input, mu, log_var]
        recons = self.decode(z)
        if self.verbose: print(f"self.decode(z) {recons.shape}")
        recons = rearrange(recons, 'b c f h w -> b (c f) h w')
        if self.verbose: print(f"rearrange(recons, 'b c f h w -> b (c f) h w') {recons.shape}")
        return  [recons, mu, log_var]
