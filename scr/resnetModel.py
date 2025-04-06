import torch
import torch.nn as nn
import torch.nn.functional as F

# code implementation for ResNet (ResNet-18 and ResNet-34)

# Residual block for ResNet-18 and ResNet-34
class ResidualBlock(nn.Module):
    """
    Implements a residual block for ResNet-18 and ResNet-34.

    This block consists of two convolutional layers with batch normalization and ReLU activation.
    It includes a skip connection that allows the gradient to flow directly, mitigating the vanishing 
    gradient problem. If necessary, a downsampling layer is added to match dimensions.

    Attributes:
        expansion (int): Expansion factor for the number of output channels.
        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization for the first convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization for the second convolutional layer.
        relu (nn.ReLU): ReLU activation function.
        downsample (nn.Sequential, optional): Downsampling layer for dimensionality matching.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride for the first convolutional layer. Default is 1.
        is_first_block (bool, optional): Indicates whether the block is the first in its layer, 
                                         requiring downsampling. Default is False.
    """
    expansion = 1
    def __init__(self, in_channels, out_channels,
                 stride=1, is_first_block=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        # Skip connection goes through 1x1 convolution with stride=2 for 
        # the first blocks of conv3_x, conv4_x, and conv5_x layers for matching
        # spatial dimension of feature maps and number of channels in order to 
        # perform the add operations.
        self.downsample = None
        if is_first_block and stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                      out_channels=out_channels,
                                                      kernel_size=1,
                                                      stride=stride,
                                                      padding=0),
                                            nn.BatchNorm2d(out_channels))
            
    def forward(self, X):
        """
        Forward pass through the residual block.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through the residual block.
        """
        identity = X.clone()
        X = self.relu(self.bn1(self.conv1(X)))
        X = self.bn2(self.conv2(X))

        if self.downsample:
            identity = self.downsample(identity)
        X += identity
        X = self.relu(X)

        return X

class ResNet(nn.Module):
    """
    Implements the ResNet architecture (ResNet-18 and ResNet-34 variants).

    This class constructs a ResNet model with configurable depth using ResidualBlocks.
    It includes an initial convolution layer, multiple residual layers, an adaptive pooling layer,
    and a fully connected classification layer.

    Attributes:
        conv1 (nn.Sequential): Initial convolutional layer with batch normalization and pooling.
        conv2_x (nn.Sequential): First residual layer.
        conv3_x (nn.Sequential): Second residual layer.
        conv4_x (nn.Sequential): Third residual layer.
        conv5_x (nn.Sequential): Fourth residual layer.
        avgpool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        fc (nn.Linear): Fully connected layer for classification.

    Args:
        ResBlock (nn.Module): Residual block to use (e.g., `ResidualBlock`).
        n_classes (int): Number of output classes.
        n_blocks_list (list of int, optional): Number of blocks in each layer. Default is [2, 2, 2, 2].
        out_channels_list (list of int, optional): Number of output channels in each layer. Default is [64, 128, 256, 512].
        num_channels (int, optional): Number of input channels (e.g., 3 for RGB images). Default is 3.
    """
    def __init__(self, ResBlock, n_classes, n_blocks_list=[2, 2, 2, 2], 
                        out_channels_list=[64, 128, 256, 512], num_channels=3):
        super(ResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=num_channels,
                                             out_channels=64, kernel_size=7,
                                             stride=2, padding=3),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)         
                                    )
        
        # Second layer
        in_channels = 64
        # First block of second layer do not require downsampling
        self.conv2_x = self.CreateLayer(ResBlock, n_blocks_list[0], 
                                        in_channels, out_channels_list[0], stride=1)
        
        # Third, Forth, and Fifth layer
        # For the first blocks of conv3_x to conv5_x layers, perform downsampling using stride=2
        # ResBlock.expansion = 1
        self.conv3_x = self.CreateLayer(ResBlock, n_blocks_list[1],
                                        out_channels_list[0]*ResBlock.expansion,
                                        out_channels_list[1], stride=2)
        self.conv4_x = self.CreateLayer(ResBlock, n_blocks_list[2],
                                        out_channels_list[1]*ResBlock.expansion,
                                        out_channels_list[2], stride=2)
        self.conv5_x = self.CreateLayer(ResBlock, n_blocks_list[3],
                                        out_channels_list[2]*ResBlock.expansion,
                                        out_channels_list[3], stride=2)

        # Average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Linear layer
        self.fc = nn.Linear(out_channels_list[3]*ResBlock.expansion, n_classes)

    def forward(self, X):
        """
        Forward pass through the ResNet model.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, n_classes).
        """
        X = self.conv1(X)
        X = self.conv2_x(X)
        X = self.conv3_x(X)
        X = self.conv4_x(X)
        X = self.conv5_x(X)

        X = self.avgpool(X)
        X = X.reshape(X.shape[0], -1)
        out = self.fc(X)

        return out
    
    def CreateLayer(self, ResBlock, n_blocks, in_channels, out_channels, stride=1):
        """
        Creates a sequence of residual blocks to form a ResNet layer.

        Args:
            ResBlock (nn.Module): Residual block class.
            n_blocks (int): Number of residual blocks in this layer.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride for the first block. Default is 1.

        Returns:
            nn.Sequential: Sequential container of residual blocks forming the layer.
        """
        layer = []
        for i in range(n_blocks):
            if i == 0:
                layer.append(ResBlock(in_channels, out_channels, 
                                      stride=stride, is_first_block=True))
            else:
                # ResBlock.expansion = 1
                layer.append(ResBlock(out_channels*ResBlock.expansion, out_channels))

        return nn.Sequential(*layer)


if __name__ == "__main__":

    sample = ResNet(ResidualBlock, 3, [2,2,2,2], [64,128,256,512], 3)

    x = torch.rand(1, 3, 64, 64)
    outputs = sample(x)
    print(outputs.shape)
