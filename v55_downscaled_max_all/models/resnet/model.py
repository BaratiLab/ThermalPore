# model.py
# https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py

import torch.nn as  nn

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, verbose=False):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm3d(out_channels)
        
        self.conv3 = nn.Conv3d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm3d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        self.verbose = verbose
        
    def forward(self, x):
        identity = x.clone()
        if self.verbose: print(f"identity = x.clone() {x.shape}")
        x = self.conv1(x)
        if self.verbose: print(f"x = self.conv1(x) {x.shape}")
        x = self.batch_norm1(x)
        if self.verbose: print(f"x = self.batch_norm1(x) {x.shape}")
        x = self.relu(x)
        if self.verbose: print(f"x = self.relu() {x.shape}")

        x = self.conv2(x)
        if self.verbose: print(f"x = self.conv2(x) {x.shape}")
        x = self.batch_norm2(x)
        if self.verbose: print(f"x = self.batch_norm2(x) {x.shape}")
        x = self.relu(x)
        if self.verbose: print(f"x = self.relu() {x.shape}")
        
        x = self.conv3(x)
        if self.verbose: print(f"x = self.conv3(x) {x.shape}")
        x = self.batch_norm3(x)
        if self.verbose: print(f"x = self.batch_norm3(x) {x.shape}")
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
            if self.verbose:
                print(f"identity = self.i_downsample(identity) {identity.shape}")
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


        
        
class Model(nn.Module):
    def __init__(self, ResBlock, layer_list, name, num_classes, num_channels=3, verbose=False):
        super(Model, self).__init__()
        self.verbose = verbose
        self.in_channels = 64
        
        self.conv1 = nn.Conv3d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool3d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        if self.verbose: print(f"x {x.shape}")
        x = self.conv1(x)
        if self.verbose: print(f"x = self.conv1(x) {x.shape}")
        x = self.batch_norm1(x)
        if self.verbose: print(f"x = self.batch_norm1(x) {x.shape}")
        x = self.relu(x)
        if self.verbose: print(f"x = self.relu(x) {x.shape}")
        x = self.max_pool(x)
        if self.verbose: print(f"x = self.max_pool(x) {x.shape}")

        x = self.layer1(x)
        if self.verbose: print(f"x = self.layer1(x) {x.shape}")
        x = self.layer2(x)
        if self.verbose: print(f"x = self.layer2(x) {x.shape}")
        x = self.layer3(x)
        if self.verbose: print(f"x = self.layer3(x) {x.shape}")
        x = self.layer4(x)
        if self.verbose: print(f"x = self.layer4(x) {x.shape}")
        
        x = self.avgpool(x)
        if self.verbose: print(f"x = self.avgpool(x) {x.shape}")
        x = x.reshape(x.shape[0], -1)
        if self.verbose: print(f"x = x.reshape(x.shape[0], -1) {x.shape}")
        x = self.fc(x)
        if self.verbose: print(f"x = self.fc(x) {x.shape}")

        # Reshape to label
        x = x.reshape(x.shape[0], 64, 64, 9)
        if self.verbose: print(f"x = x.reshape(x.shape[0], 64, 64, 9) {x.shape}")

        x = self.sigmoid(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm3d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride, verbose = self.verbose))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)
        
# def ResNet50(num_classes, channels=3):
#     return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
    
# def ResNet101(num_classes, channels=3):
#     return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

# def ResNet152(num_classes, channels=3):
#     return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)
