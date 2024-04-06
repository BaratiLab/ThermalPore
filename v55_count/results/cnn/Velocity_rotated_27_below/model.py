# model.py

from torch import nn
from einops import rearrange

class Model(nn.Module):
  def __init__(
      self,
      frames,
      conv_kernel_size,
      conv_padding,
      conv_stride,
      max_pool_kernel_size,
      max_pool_stride,
      name,
      out_channels,
      image_size,
      verbose = False
    ):
    super().__init__()
    self.image_size = image_size
    self.frames = frames
    self.conv_output_features = (self.frames//16)*(self.image_size**2)
    self.verbose = verbose
    self.conv1 = nn.Conv3d(
      in_channels = frames,
      out_channels = frames // 2,
      kernel_size = conv_kernel_size,
      padding = conv_padding,
      stride = conv_stride
    )
    self.conv1_bn = nn.BatchNorm3d(frames // 2)

    self.conv2 = nn.Conv3d(
      in_channels = frames // 2,
      out_channels = frames // 4,
      kernel_size = conv_kernel_size,
      padding = conv_padding,
      stride = conv_stride
    )
    self.conv2_bn = nn.BatchNorm3d(frames // 4)

    self.conv3 = nn.Conv3d(
      in_channels = frames // 4,
      out_channels = frames // 8,
      kernel_size = conv_kernel_size,
      padding = conv_padding,
      stride = conv_stride
    )
    self.conv3_bn = nn.BatchNorm3d(frames // 8)

    self.conv4 = nn.Conv3d(
      in_channels = frames // 8,
      out_channels = frames // 16,
      kernel_size = conv_kernel_size,
      padding = conv_padding,
      stride = conv_stride
    )
    self.conv4_bn = nn.BatchNorm3d(frames // 16)
    self.fc1 = nn.Linear(self.conv_output_features, self.conv_output_features //8)
    self.fc2 = nn.Linear(self.conv_output_features//8, out_channels)

    self.max_pool = nn.MaxPool3d(
      kernel_size = max_pool_kernel_size,
      stride = max_pool_stride
    )
    self.relu = nn.ReLU()
    # self.sigmoid = nn.Sigmoid()
  
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
    x = x.view(-1, self.conv_output_features)
    if self.verbose: print(f"x = self.view(x) {x.shape}")
    x = self.fc1(x)
    if self.verbose: print(f"x = self.fc1(x) {x.shape}")
    x = self.fc2(x)
    if self.verbose: print(f"x = self.fc2(x) {x.shape}")
    # x = self.max_pool(x)
    # if self.verbose: print(f"x = self.max_pool(x) {x.shape}")

    # x = rearrange(x, 'b f c h w -> (b c) h w f')
    # if self.verbose: print(f"x = rearrange(x, 'b f c h w -> (b c) h w f') {x.shape}")
    x = x.flatten()
    if self.verbose: print(f"x = x.flatten() {x.shape}")

    x = self.relu(x)
    # x = self.sigmoid(x)

    return x
