# model.py

from torch import nn
from einops import rearrange

class Model(nn.Module):
  def __init__(
      self,
      frames,
      conv_kernel_size,
      conv_padding,
      max_pool_kernel_size,
      max_pool_stride,
      name,
      out_channels,
      verbose = False
    ):
    super().__init__()
    self.verbose = verbose
    self.conv1 = nn.Conv3d(
      in_channels = frames,
      out_channels = frames // 2,
      kernel_size = conv_kernel_size,
      padding = conv_padding
    )
    self.conv1_bn = nn.BatchNorm3d(frames // 2)

    self.conv2 = nn.Conv3d(
      in_channels = frames // 2,
      out_channels = frames // 4,
      kernel_size = conv_kernel_size,
      padding = conv_padding
    )
    self.conv2_bn = nn.BatchNorm3d(frames // 4)

    self.conv3 = nn.Conv3d(
      in_channels = frames // 4,
      out_channels = frames // 8,
      kernel_size = conv_kernel_size,
      padding = conv_padding
    )
    self.conv3_bn = nn.BatchNorm3d(frames // 8)

    self.conv4 = nn.Conv3d(
      in_channels = frames // 8,
      out_channels = out_channels,
      kernel_size = conv_kernel_size,
      padding = conv_padding
    )
    self.conv4_bn = nn.BatchNorm3d(out_channels)

    self.max_pool = nn.MaxPool3d(
      kernel_size = max_pool_kernel_size,
      stride = max_pool_stride
    )
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, video):
    if self.verbose: print(f"video.shape {video.shape}")

    x = rearrange(video, 'b c f h w -> b f c h w')
    if self.verbose: print(f"x = rearrange(video, 'b c f h w -> b f c h w') {x.shape}")

    x = self.conv1(x)
    if self.verbose: print(f"x = self.conv1(x) {x.shape}")
    x = self.conv1_bn(x)
    if self.verbose: print(f"x = self.conv1_bn(x) {x.shape}")
    x = self.relu(x)
    if self.verbose: print(f"x = self.relu(x) {x.shape}")
    # x = self.max_pool(x)
    # if self.verbose: print(f"x = self.max_pool(x) {x.shape}")

    x = self.conv2(x)
    if self.verbose: print(f"x = self.conv2(x) {x.shape}")
    x = self.conv2_bn(x)
    if self.verbose: print(f"x = self.conv2_bn(x) {x.shape}")
    x = self.relu(x)
    if self.verbose: print(f"x = self.relu(x) {x.shape}")
    # x = self.max_pool(x)
    # if self.verbose: print(f"x = self.max_pool(x) {x.shape}")

    x = self.conv3(x)
    if self.verbose: print(f"x = self.conv3(x) {x.shape}")
    x = self.conv3_bn(x)
    if self.verbose: print(f"x = self.conv3_bn(x) {x.shape}")
    x = self.relu(x)
    if self.verbose: print(f"x = self.relu(x) {x.shape}")
    # x = self.max_pool(x)
    # if self.verbose: print(f"x = self.max_pool(x) {x.shape}")

    x = self.conv4(x)
    if self.verbose: print(f"x = self.conv4(x) {x.shape}")
    x = self.conv4_bn(x)
    if self.verbose: print(f"x = self.conv4_bn(x) {x.shape}")
    x = self.relu(x)
    if self.verbose: print(f"x = self.relu(x) {x.shape}")
    # x = self.max_pool(x)
    # if self.verbose: print(f"x = self.max_pool(x) {x.shape}")

    x = rearrange(x, 'b f c h w -> (b c) h w f')
    if self.verbose: print(f"x = rearrange(x, 'b f c h w -> (b c) h w f') {x.shape}")

    x = self.sigmoid(x)

    return x
