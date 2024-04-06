# model.py

import torch
from torch import nn
import torchvision.models as models

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

resnet50 = models.resnet50

# helpers

def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # print(f"self.to_qkv(x) {self.to_qkv(x).shape}")
        # print(f"chuck len {len(self.to_qkv(x).chunk(3, dim = -1))}")
        x_length = len(x.shape)
        if (x_length == 3):
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        elif (x_length == 4):
            q, k, v = map(lambda t: rearrange(t, 'b f n (h d) -> b h f n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        if (x_length == 3):
            out = rearrange(out, 'b h n d -> b n (h d)')
        elif(x_length == 4):
            out = rearrange(out, 'b h f n d -> b f n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
    
class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        # self.conv1 = nn.Conv2d(
        #     features, features, kernel_size=3, stride=1, padding=1, bias=True
        # )

        # self.conv2 = nn.Conv2d(
        #     features, features, kernel_size=3, stride=1, padding=1, bias=True
        # )
        self.conv1 = nn.Conv3d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv3d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x
    
class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)
        # print(output.shape)

        output = nn.functional.interpolate(
            output,
            scale_factor=2,
            mode="trilinear",
            # mode="bilinear",
            align_corners=True
        )

        return output

class Model(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        dim,
        spatial_depth,
        temporal_depth,
        heads,
        mlp_dim,
        name,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        out_channels = 9,
        verbose = False,
    ):
        super().__init__()
        self.verbose = verbose
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        self.image_patch_height = image_height // patch_height 

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_image_patches = (image_height // patch_height) * (image_width // patch_width)
        num_frame_patches = (frames // frame_patch_size)

        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.global_average_pool = pool == 'mean'

        # self.to_patch_embedding = nn.Sequential(
        self.patch_rearrange = Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size)
        self.patch_layer_norm_1 = nn.LayerNorm(patch_dim)
        self.patch_linear = nn.Linear(patch_dim, dim)
        self.patch_layer_norm_2 = nn.LayerNorm(dim)
        # )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
        # self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, 1, dim)) if not self.global_average_pool else None

        self.spatial_transformer = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)
        self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.dpt_head_conv_1 = nn.Conv3d(dim, dim // 2, kernel_size = 3, stride = 1, padding = 1)
        self.dpt_head_interp = nn.functional.interpolate 
        self.dpt_head_conv_2 = nn.Conv3d(dim // 2, dim // 4, kernel_size = 3, stride = 1, padding = 1)
        self.dpt_head_conv_3 = nn.Conv3d(
            dim // 4,
            1,
            kernel_size = 3,
            stride = 1,
            # padding = 0
            padding = 1
        )
        self.dpt_head_conv_4 = nn.Conv2d(num_frame_patches * 2, out_channels=out_channels, kernel_size = 3, stride = 2, padding = 1)
        self.relu = nn.ReLU(True)

        # self.bn_1 = nn.BatchNorm3d(dim)
        # self.cnn_1 = nn.Conv3d(in_channels=dim, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        # self.cnn_2 = nn.Conv3d(in_channels=num_frame_patches, out_channels=9, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.sigmoid = nn.Sigmoid()
        # self.scratch = _make_scratch([8, 16, 32, 64], 64)

        self.reassemble_1 = nn.Conv3d(in_channels = dim, out_channels = dim, kernel_size=3, stride=1, padding=1)
        self.fusion_1 = FeatureFusionBlock(dim)


    def forward(self, video):
        if self.verbose: print(f"\nvideo {video.shape}")
        x = self.patch_rearrange(video)
        if self.verbose: print(f"self.patch_rearrange(x) {x.shape}")

        x = self.patch_layer_norm_1(x)
        if self.verbose: print(f"self.patch_layer_norm_1(x) {x.shape}")

        x = self.patch_linear(x)
        if self.verbose: print(f"self.patch_linear(x) {x.shape}")

        x = self.patch_layer_norm_2(x)
        if self.verbose: print(f"self.patch_layer_norm_2(x) {x.shape}")

        if self.verbose: print(f"self.to_patch_embedding(video) {x.shape}")
        b, f, n, _ = x.shape

        if self.verbose: print(f"pos_embedding {self.pos_embedding[:, :f, :n].shape}")

        x = x + self.pos_embedding[:, :f, :n]

        if self.verbose: print(f"x + self.pos_embedding[:, :f, :n] {x.shape}")
        if exists(self.spatial_cls_token):
            spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 1 d', b = b, f = f)
            if self.verbose: print(f"spatial_cls_tokens {spatial_cls_tokens.shape}")
            x = torch.cat((spatial_cls_tokens, x), dim = 2)
            if self.verbose: print(f"torch.cat((spatial_cls_tokens, x), dim = 2) {x.shape}")

        x = self.dropout(x)

        x = rearrange(x, 'b f n d -> (b f) n d')

        if self.verbose: print(f"rearrange {x.shape}")

        # attend across space

        x = self.spatial_transformer(x)

        if self.verbose: print(f"self.spatial_transformer(x) {x.shape}")

        x = rearrange(x, '(b f) n d -> b f n d', b = b)
        if self.verbose: print(f"rearrange {x.shape}")

        # Remove Spatial Class Token
        patch_dimension = x.shape[2] - 1
        if self.verbose: print(patch_dimension)
        x = x[:, :, :patch_dimension, :]
        if self.verbose: print(f"Remove Spatial Class Token {x.shape}")

        # # excise out the spatial cls tokens or average pool for temporal attention
        # x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean')
        if self.verbose: print(f"reduce {x.shape}")

        # append temporal CLS tokens
        if exists(self.temporal_cls_token):
            # temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b = b)
            temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 1 d-> b 1 pd d', b = b, pd = patch_dimension)
            if self.verbose: print(f"temporal_cls_tokens {temporal_cls_tokens.shape}")

            x = torch.cat((temporal_cls_tokens, x), dim = 1)
            if self.verbose: print(f"x {x.shape}")

        # attend across time

        x = self.temporal_transformer(x)

        if self.verbose: print(f"self.temporal_transformer(x) {x.shape}")

        # Remove Temporal Class Token
        frame_dimension = x.shape[1] - 1
        if self.verbose: print(frame_dimension)
        x = x[:, :frame_dimension, :, :]
        if self.verbose: print(f"Remove Temporal Class Token {x.shape}")

        # excise out temporal cls token or average pool
        # x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')
        # if self.verbose: print(f"reduce {x.shape}")

        # x = self.to_latent(x)
        # if self.verbose: print(f"x = self.to_latent(x) {x.shape}")

        x = rearrange(x, 'b f (h w) d -> b d h w f', h = self.image_patch_height)
        if self.verbose: print(f"rearrange(x, 'b f (h w) d -> b d h w f') {x.shape}")
        
        x = self.reassemble_1(x)
        if self.verbose: print(f"self.reassemble_1(x) {x.shape}")

        x = self.fusion_1(x)
        if self.verbose: print(f"self.fusion_1(x) {x.shape}")

        x = self.dpt_head_conv_1(x)
        if self.verbose: print(f"self.dpt_head_conv_1(x) {x.shape}")
        # x = self.dpt_head_interp(x, scale_factor=2, mode="trilinear", align_corners=True)
        # if self.verbose: print(f"self.dpt_head_interp(x) {x.shape}")
        x = self.dpt_head_conv_2(x)
        if self.verbose: print(f"self.dpt_head_conv_2(x) {x.shape}")
        x = self.relu(x)
        if self.verbose: print(f"relu(x) {x.shape}")
        x = self.dpt_head_conv_3(x)
        if self.verbose: print(f"self.dpt_head_conv_3(x) {x.shape}")
        x = self.relu(x)
        if self.verbose: print(f"relu(x) {x.shape}")

        x = rearrange(x, 'b d h w f -> (b d) f h w')
        if self.verbose: print(f"rearrange(x, 'b d h w f -> (b d) f h w') {x.shape}")

        x = self.dpt_head_conv_4(x)
        if self.verbose: print(f"self.dpt_head_conv_4(x) {x.shape}")

        x = rearrange(x, 'b f h w -> b h w f')
        if self.verbose: print(f"rearrange(x, 'b f h w -> b h w f') {x.shape}")

        # x = self.bn_1(x)

        # # Apply CNN layers
        # x = self.cnn_1(x)
        # if self.verbose: print(f"x = self.cnn_1(x) {x.shape}")
        # x = self.relu(x)

        # x = rearrange(x, 'b d h w f -> (b d) f h w')
        # if self.verbose: print(f"rearrange(x, 'b d h w f -> (b d) f h w') {x.shape}")

        # x = nn.functional.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)
        # if self.verbose: print(f"interpolate {x.shape}")

        # x = rearrange(x, 'b f h w -> b f 1 h w')
        # if self.verbose: print(f"rearrange(x, 'b f h w -> b f 1 h w') {x.shape}")

        # x = self.cnn_2(x)
        # if self.verbose: print(f"x = self.cnn_2(x) {x.shape}")
        # # x = self.relu(x)

        # x = rearrange(x, 'b f 1 h w -> b h w (f 1) ')
        # if self.verbose: print(f"rearrange(x, 'b f 1 h w -> b h w (f 1)') {x.shape}")

        x = self.sigmoid(x)

        # # x = self.refinement_1(x) 
        # if self.verbose: print(f"self.refinement_1(x) {x.shape}")

        # Apply DPT
        return x 