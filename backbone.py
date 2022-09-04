# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import cv2
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from configs import HyperParams
from torchvision import utils as vutils
# Basic ResNet model
from utils import batch_augment

EPSILON = 1e-12
def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)



# Simple Conv Block
class ConvBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, pool = True, padding = 1):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim

        self.C      = nn.Conv2d(indim, outdim, 3, padding= padding)
        self.BN     = nn.BatchNorm2d(outdim)
        self.relu   = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool   = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        # for layer in self.parametrized_layers:
        #     init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)


    def forward(self,x):
        out = self.trunk(x)
        return out



class ResBlock(nn.Module):
	def __init__(self, nFin, nFout):
		super(ResBlock, self).__init__()

		self.conv_block = nn.Sequential()
		self.conv_block.add_module('BNorm1', nn.BatchNorm2d(nFin))
		self.conv_block.add_module('LRelu1', nn.LeakyReLU(0.2))
		self.conv_block.add_module('ConvL1', nn.Conv2d(nFin,  nFout, kernel_size=3, padding=1, bias=False))
		self.conv_block.add_module('BNorm2', nn.BatchNorm2d(nFout))
		self.conv_block.add_module('LRelu2', nn.LeakyReLU(0.2))
		self.conv_block.add_module('ConvL2', nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))
		self.conv_block.add_module('BNorm3', nn.BatchNorm2d(nFout))
		self.conv_block.add_module('LRelu3', nn.LeakyReLU(0.2))
		self.conv_block.add_module('ConvL3', nn.Conv2d(nFout, nFout, kernel_size=3, padding=1, bias=False))

		self.skip_layer = nn.Conv2d(nFin, nFout, kernel_size=1, stride=1)


	def forward(self, x):
		return self.skip_layer(x) + self.conv_block(x)


class ConvNetNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth):
        super(ConvNetNopool,self).__init__()
        trunk = []
        for i in range(depth):   #depth =4
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1  ) #only first two layer has pooling and no padding
            trunk.append(B)

        # self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64,19,19]

        self.M = 16
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        self.n_way = 5
        self.n_support = 1
        self.n_query = 16
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_se = nn.Sequential(
            nn.Linear(16, 16 // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(16 // 4, 16, bias=False),
            nn.Sigmoid()
        )
        self.fc_se1 = nn.Sequential(
            nn.Linear(64, 64 // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64 // 16, 64, bias=False),
            nn.Sigmoid()
        )
        self.attention = BasicConv2d(64, 16, kernel_size=1)
        self.bap = BAP(pool='GAP')

        self.sigmoid = nn.Sigmoid()
        self.LocalMaxGlobalMin = LocalMaxGlobalMin(rho=1,nchannels=64)
        self.out_planes = [64, 96, 128, 256]
        self.num_stages = 4
        num_planes = [self.out_planes[0], ] + self.out_planes

        self.feat_extractor = nn.Sequential()
        self.feat_extractor.add_module('ConvL0', nn.Conv2d(3, num_planes[0], kernel_size=3, padding=1))

        for i in range(self.num_stages):
            self.feat_extractor.add_module('ResBlock' + str(i), ResBlock(num_planes[i], num_planes[i + 1]))
            if i < self.num_stages - 2:
                self.feat_extractor.add_module('MaxPool' + str(i), nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.feat_extractor.add_module('ReluF1', nn.LeakyReLU(0.2, True))  # get Batch*256*21*21

    def att_map(self,x):
        attention_maps = self.attention(x)
        b, c, _, _ = attention_maps.size()  #b=85 c=16  torch.Size([85, 16, 19, 19])
        y = self.avg_pool(attention_maps).view(b, c) #y torch.Size([85, 16])
        y = self.fc_se(y).view(b, c, 1, 1)
        attention_maps = attention_maps * y.expand_as(attention_maps)  #attention torch.Size([85, 16, 19, 19])

        feature_matrix = self.bap(x, attention_maps, 1) #torch.Size([85, 1024])
        return feature_matrix
    def feature(self,x):
        out0 = self.layer1(x)
        out1 = self.layer2(out0)
        out2 = self.layer3(out1)
        out3 = self.layer4(out2)
        return out3
    def localfea(self, x):
        b,c,_,_ = x.size()
        x1 = self.avg_pool(x).view(b,c)
        x1 = self.fc_se1(x1).view(b,c,1,1)
        x1 = x * x1.expand_as(x)

        return x1
    def forward(self,x):
        out = self.feature(x)
        feature_matrix = self.att_map(out)  # torch.Size([85, 1024])
        feature_matrix = feature_matrix.view(-1, self.M, self.final_feat_dim[0])  # torch.Size([85, 16, 64])
        return out,feature_matrix



class LocalMaxGlobalMin(nn.Module):

    def __init__(self, rho, nchannels, nparts=2, device='cpu'):
        super(LocalMaxGlobalMin, self).__init__()
        self.nparts = nparts
        self.device = device
        self.nchannels = nchannels
        self.rho = rho

        nlocal_channels_norm = nchannels // self.nparts
        reminder = nchannels % self.nparts
        nlocal_channels_last = nlocal_channels_norm
        if reminder != 0:
            nlocal_channels_last = nlocal_channels_norm + reminder

        # seps records the indices partitioning feature channels into separate parts
        seps = []
        sep_node = 0
        for i in range(self.nparts):
            if i != self.nparts - 1:
                sep_node += nlocal_channels_norm
                # seps.append(sep_node)
            else:
                sep_node += nlocal_channels_last
            seps.append(sep_node)
        self.seps = seps

    def forward(self, x):
        x = x.pow(2)
        intra_x = []
        inter_x = []
        for i in range(self.nparts):
            if i == 0:
                intra_x.append((1 - x[:, :self.seps[i], :self.seps[i]]).mean())
            else:
                intra_x.append((1 - x[:, self.seps[i - 1]:self.seps[i], self.seps[i - 1]:self.seps[i]]).mean())
                inter_x.append(x[:, self.seps[i - 1]:self.seps[i], :self.seps[i - 1]].mean())
        try:
            loss = self.rho * 0.5 * (sum(intra_x) / self.nparts + sum(inter_x) / (self.nparts * (self.nparts - 1) / 2))
        except:
            print('被除数为0！！！')
        return loss


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, tensor):
        loss_sum = torch.zeros(1).cuda()
        indexes = Loss.get_max_index(tensor)
        for i in range(len(indexes)):
            max_x, max_y = indexes[i]
            for j in range(tensor.size(2)):
                for k in range(tensor.size(3)):
                    loss_sum += ((max_x - j) * (max_x - j) + (max_y - k) * (max_y - k)) * tensor[i, j, k]
            return loss_sum

    @staticmethod
    def get_max_index(tensor):
        shape = tensor.shape #torch.Size([85, 16, 19, 19])
        indexes = []
        for e in range(shape[0]):
            for i in range(shape[1]):
                mx = tensor[e,i, 0, 0]
                x, y = 0, 0
                for j in range(shape[2]):
                    for k in range(shape[3]):
                        if tensor[e,i, j, k] > mx:
                            mx = tensor[e,i, j, k]
                            x, y = j, k
                indexes.append([x, y])
        return indexes
loss_fn = Loss()
def Conv4NP():
    return ConvNetNopool(4)

class ChannelGate(nn.Module):
    """generation channel attention mask"""
    def __init__(self, out_channels):
        super(ChannelGate, self).__init__()
        #---change--- dilation
        self.conv1 = nn.Conv2d(out_channels,out_channels//16,kernel_size=1,stride=1,padding=0,dilation=3)
        self.conv2 = nn.Conv2d(out_channels//16,out_channels,kernel_size=1,stride=1,padding=0,dilation=3)
    def forward(self, x):
        x = nn.AdaptiveAvgPool2d(output_size=1)(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = torch.sigmoid(self.conv2(x))
        return x



class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(1024,64,kernel_size=1,padding=0)
    def forward(self, features, attentions,flag):
        B, C, H, W = features.size() # 85 * 64 * 19 * 19
        _, M, AH, AW = attentions.size() # 85 * 16 * 19 * 19
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix




