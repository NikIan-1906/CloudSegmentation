import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


test_transform = A.Compose([A.Resize(300, 300),
                           A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255),
                           ToTensorV2()])

def double_conv(in_ch, out_ch):
    conv = nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

    return conv


def padder(left_tensor, right_tensor):
    # left_tensor is the tensor on the encoder side of UNET
    # right_tensor is the tensor on the decoder side  of the UNET

    if left_tensor.shape != right_tensor.shape:
        padded = torch.zeros(left_tensor.shape)
        padded[:, :, :right_tensor.shape[2], :right_tensor.shape[3]] = right_tensor
        return padded.to(device)

    return right_tensor.to(device)


class UNET(nn.Module):
    def __init__(self, in_chnls, n_classes):
        super(UNET, self).__init__()

        self.in_chnls = in_chnls
        self.n_classes = n_classes

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv_1 = double_conv(in_ch=self.in_chnls, out_ch=64)
        self.down_conv_2 = double_conv(in_ch=64, out_ch=128)
        self.down_conv_3 = double_conv(in_ch=128, out_ch=256)
        self.down_conv_4 = double_conv(in_ch=256, out_ch=512)
        self.down_conv_5 = double_conv(in_ch=512, out_ch=1024)
        # print(self.down_conv_1)

        self.up_conv_trans_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_trans_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_trans_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_trans_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.up_conv_1 = double_conv(in_ch=1024, out_ch=512)
        self.up_conv_2 = double_conv(in_ch=512, out_ch=256)
        self.up_conv_3 = double_conv(in_ch=256, out_ch=128)
        self.up_conv_4 = double_conv(in_ch=128, out_ch=64)

        self.conv_1x1 = nn.Conv2d(in_channels=64, out_channels=self.n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        # encoding
        x1 = self.down_conv_1(x)
        p1 = self.max_pool(x1)
        x2 = self.down_conv_2(p1)
        p2 = self.max_pool(x2)
        x3 = self.down_conv_3(p2)
        p3 = self.max_pool(x3)
        x4 = self.down_conv_4(p3)
        p4 = self.max_pool(x4)
        x5 = self.down_conv_5(p4)

        # decoding
        d1 = self.up_conv_trans_1(x5)  # up transpose convolution ("up sampling" as called in UNET paper)
        pad1 = padder(x4, d1)  # padding d1 to match x4 shape
        cat1 = torch.cat([x4, pad1],
                         dim=1)  # concatenating padded d1 and x4 on channel dimension(dim 1) [batch(dim 0),channel(dim 1),height(dim 2),width(dim 3)]
        uc1 = self.up_conv_1(cat1)  # 1st up double convolution

        d2 = self.up_conv_trans_2(uc1)
        pad2 = padder(x3, d2)
        cat2 = torch.cat([x3, pad2], dim=1)
        uc2 = self.up_conv_2(cat2)

        d3 = self.up_conv_trans_3(uc2)
        pad3 = padder(x2, d3)
        cat3 = torch.cat([x2, pad3], dim=1)
        uc3 = self.up_conv_3(cat3)

        d4 = self.up_conv_trans_4(uc3)
        pad4 = padder(x1, d4)
        cat4 = torch.cat([x1, pad4], dim=1)
        uc4 = self.up_conv_4(cat4)

        conv_1x1 = self.conv_1x1(uc4)
        return conv_1x1
        # print(conv_1x1.shape)