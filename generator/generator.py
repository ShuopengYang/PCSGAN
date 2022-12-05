import torch
from torch import nn as nn
from torch.nn import functional as F

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ChannelAttention(nn.Module):
    """Channel attention.
    Args:
        num_feat (int): Channel number of intermediate numerical features.
    """
    def __init__(self, num_feat=64):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):
    def __init__(self, num_feat=64):
        super(CAB, self).__init__()
        
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            ChannelAttention(num_feat)
            )

    def forward(self, x):
        return self.cab(x)

class CombinationUnit(nn.Module):
    """Combination unit of Residual Dense Block (RDB) and Residual Channel Attention Block (RCAB).
    Args:
        num_feat (int): Channel number of intermediate numerical features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat = 64, num_grow_ch = 9):
        super(CombinationUnit, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_grow_ch, 3, 1, 1)   
        self.conv6 = nn.Conv2d(num_feat + 5 * num_grow_ch, num_feat, 3, 1, 1)   
        self.attention = CAB(64, 1, 1)    
        self.lrelu = nn.LeakyReLU(negative_slope = 0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x)) 
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1))) 
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.lrelu(self.conv5(torch.cat((x, x1, x2, x3, x4), 1)))
        x6 = self.conv6(torch.cat((x, x1, x2, x3, x4, x5), 1))
        x7 = x6 * 0.2 + x 
        x8 = self.attention(x7)
        return x8 * 0.2 + x7

class ResidualChannelAttentionDenseBlock(nn.Module):
    """Residual Channel Attention Dense Block.
    Args: 
        num_feat (int): Channel number of intermediate numerical features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch = 9):
        super(ResidualChannelAttentionDenseBlock, self).__init__()
        self.rdb1 = CombinationUnit(num_feat, num_grow_ch)
        self.rdb2 = CombinationUnit(num_feat, num_grow_ch)
        self.rdb3 = CombinationUnit(num_feat, num_grow_ch)   
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = self.conv(out)
        return out * 0.2 + x

class generatorNet(nn.Module):
    """Generator network of PCSGAN. 
    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate numerical features.
            Default: 64
        num_block (int): Number of RCADBs. Defaults: 6
        num_grow_ch (int): Channels for each growth. Default: 9.
    """

    def __init__(self, num_in_ch, num_out_ch, num_feat = 64, num_block = 6, num_grow_ch = 9):
        super(generatorNet, self).__init__()
 
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualChannelAttentionDenseBlock, num_block, num_feat = num_feat, num_grow_ch = num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.conv1 = nn.Conv2d(2, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope = 0.2, inplace = True)

    def forward(self, x, K_fine):
        feat = x
        """shallow numerical feature extraction"""
        feat = self.conv_first(feat)
        """deep numerical feature extraction"""
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        """numerical reconstruction"""
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor = 2, mode = 'nearest'))) 
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor = 2, mode = 'nearest')))
        """use the petrophysical grid data (K_fine) to constrain and guide the enhancement procedure to improve its accuracy"""
        feat = torch.cat([feat, K_fine], dim = 1)  
        feat = self.lrelu(self.conv1(feat)) 
        feat = self.lrelu(self.conv2(feat))
        
        out = self.conv_last(self.lrelu(self.conv3(feat)))
        return out