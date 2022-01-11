from torch import nn
import torch.nn.functional as F


def compute_conv_dim(dim_size, kernel_size_conv, padding_conv, stride_conv):
    return int((dim_size - kernel_size_conv + 2 * padding_conv) / stride_conv + 1)

height = 28
width = 28
channels = 1

kernel_size_conv1 = 3
padding_conv1 = 1
stride_conv1 = 1
conv1_out_channels = 16

kernel_size_conv2 = 3
padding_conv2 = 1
stride_conv2 = 1
conv2_out_channels = 32

kernel_size_conv3 = 3
padding_conv3 = 1
stride_conv3 = 1
conv3_out_channels = 64

maxkernel = 3
maxstride = 1
maxpadding = 1


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=channels,
                        out_channels = conv1_out_channels,
                        kernel_size = kernel_size_conv1,
                        stride = stride_conv1,
                        padding = padding_conv1)
        
        self.conv_out_height1 = compute_conv_dim(height, kernel_size_conv1, padding_conv1, stride_conv1)
        self.conv_out_width1 = compute_conv_dim(width, kernel_size_conv1, padding_conv1, stride_conv1)
        
        self.conv2 = nn.Conv2d(in_channels=conv1_out_channels,
                          out_channels = conv2_out_channels,
                          kernel_size = kernel_size_conv2,
                          stride = stride_conv2,
                          padding = padding_conv2)
        
        self.conv_out_height2 = compute_conv_dim(self.conv_out_height1, maxkernel, maxpadding, maxstride)
        self.conv_out_width2 = compute_conv_dim(self.conv_out_width1, maxkernel, maxpadding, maxstride)

        self.conv3 = nn.Conv2d(in_channels=conv2_out_channels,
                          out_channels = conv3_out_channels,
                          kernel_size = kernel_size_conv3,
                          stride = stride_conv3,
                          padding = padding_conv3)
        
        self.conv_out_height3 = compute_conv_dim(self.conv_out_height1, maxkernel, maxpadding, maxstride)
        self.conv_out_width3 = compute_conv_dim(self.conv_out_width1, maxkernel, maxpadding, maxstride)
        
        self.l1_in_features = conv3_out_channels * self.conv_out_height3 * self.conv_out_width3

        self.maxpool = nn.MaxPool2d(maxkernel, maxstride, padding=maxpadding)

        self.out = nn.Linear(in_features=self.l1_in_features, 
                    out_features=10,
                    bias=True)

        self.dropout = nn.Dropout(p = 0.2)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))

        x = x.view(-1, self.l1_in_features)
        
        return F.softmax(self.out(x), dim=1)