import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='instance'):
        """
        A residual block with 3D convolutions and an option to use
        BatchNorm or InstanceNorm.

        Parameters:
        -----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        norm_type: str
            Type of normalization to use ('batch' for BatchNorm, 'instance'
            for InstanceNorm).
        """
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1)

        if norm_type == 'batch':
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.bn2 = nn.BatchNorm3d(out_channels)
        elif norm_type == 'instance':
            self.bn1 = nn.InstanceNorm3d(out_channels)
            self.bn2 = nn.InstanceNorm3d(out_channels)
        else:
            raise ValueError(f"Invalid norm_type {norm_type}. Choose either "
                             "'batch' or 'instance'.")
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        identity = self.shortcut(x)

        # First Conv => Norm => Activation
        x = self.relu(self.bn1(self.conv1(x)))

        # Second Conv => Norm
        x = self.bn2(self.conv2(x))

        # Add shortcut => Activation
        x += identity
        return self.relu(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='instance'):
        super(DownBlock, self).__init__()
        self.residual = ResidualBlock(in_channels, out_channels, norm_type)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.residual(x)
        return out, self.pool(out)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='instance'):
        super(UpBlock, self).__init__()
        self.upconv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.residual = ResidualBlock(in_channels, out_channels, norm_type)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat((x, skip), dim=1)
        return self.residual(x)

# TODO: Make batchnorm stuff less redundant. Make wrapper or something


class CustomFilterUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='instance'):
        super(CustomFilterUpBlock, self).__init__()
        self.upconv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.reduce_conv = nn.Conv3d(
            out_channels * 2, out_channels, kernel_size=3, padding=1)
        if norm_type == 'instance':
            self.norm_layer = nn.InstanceNorm3d(out_channels)
        elif norm_type == 'batch':
            self.norm_layer = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.residual = ResidualBlock(out_channels, out_channels, norm_type)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat((x, skip), dim=1)
        x = self.relu(self.norm_layer(self.reduce_conv(x)))
        return self.residual(x)
