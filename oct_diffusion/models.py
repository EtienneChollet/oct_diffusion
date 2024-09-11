from torch import nn
# from modules import
from oct_diffusion.modules import DownBlock, UpBlock, ResidualBlock


# Residual U-Net
class ResUNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 norm='instance',
                 filters=[32, 64, 128, 256, 512]
                 ):

        super(ResUNet, self).__init__()
        self.filters = filters
        self.drop = nn.Dropout3d(0.2)

        # Encoder
        self.down1 = DownBlock(in_channels, self.filters[0], norm)
        self.down2 = DownBlock(self.filters[0], self.filters[1], norm)
        self.down3 = DownBlock(self.filters[1], self.filters[2], norm)
        self.down4 = DownBlock(self.filters[2], self.filters[3], norm)

        # Bottleneck
        self.bottleneck = ResidualBlock(self.filters[3], self.filters[4], norm)

        # Decoder
        self.up1 = UpBlock(self.filters[4], self.filters[3], norm)
        self.up2 = UpBlock(self.filters[3], self.filters[2], norm)
        self.up3 = UpBlock(self.filters[2], self.filters[1], norm)
        self.up4 = UpBlock(self.filters[1], self.filters[0], norm)

        # Output layer
        self.final_conv = nn.Conv3d(self.filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip1, x = self.down1(x)
        x = self.drop(x)
        skip2, x = self.down2(x)
        x = self.drop(x)
        skip3, x = self.down3(x)
        x = self.drop(x)
        skip4, x = self.down4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up1(x, skip4)
        x = self.drop(x)
        x = self.up2(x, skip3)
        x = self.drop(x)
        x = self.up3(x, skip2)
        x = self.drop(x)
        x = self.up4(x, skip1)

        # Final Convolution
        return self.final_conv(x)
