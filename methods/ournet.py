# This code is modified from https://github.com/wyharveychen/CloserLookFewShot
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
import utils

from torch.nn import functional as F
kldiv = nn.KLDivLoss(reduction='sum')


class Classifier_cosine(nn.Module):
    def __init__(self, n_way, n_query, n_support, feat_dim):
        super(Classifier_cosine, self).__init__()
        self.n_way = n_way
        self.feat_dim = feat_dim
        self.n_support = n_support

        self.layer1 = nn.Sequential(
            nn.Conv2d(feat_dim[0], feat_dim[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dim[0], momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(feat_dim[0], feat_dim[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dim[0], momentum=1, affine=True),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )

    def forward(self, z_support, z_query):
        self.n_query = z_query.size(1)  # 16

        extend_final_feat_dim = self.feat_dim.copy()  # [64, 19, 19]
        z_support = z_support.contiguous().view(self.n_way, self.n_support, *self.feat_dim).mean(1)  # [5, 64, 19, 19]

        z_query = z_query.contiguous().view(self.n_way * self.n_query, *self.feat_dim)  # [80, 64, 19, 19]

        z_support_ext = z_support.unsqueeze(0).repeat(self.n_query * self.n_way, 1, 1, 1, 1)  # [80, 5, 64, 19, 19]
        z_query_ext = z_query.unsqueeze(0).repeat(self.n_way, 1, 1, 1, 1)  # [5, 80, 64, 19, 19]
        z_query_ext = torch.transpose(z_query_ext, 0, 1)  # [80, 5, 64, 19, 19]
        z_support_ext = z_support_ext.view(-1, *extend_final_feat_dim)  # [400, 64, 19, 19]
        z_query_ext = z_query_ext.contiguous().view(-1, *extend_final_feat_dim)  # [400, 64, 19, 19]

        x_support = self.layer1(z_support_ext)  # [400, 64, 19, 19] ==> [400, 64, 9, 9]
        x_query = self.layer1(z_query_ext)  # [400, 64, 19, 19] ==> [400, 64, 9, 9]

        x_support = self.layer2(x_support)  # [400, 64, 9, 9] ==> [400, 64, 4, 4]
        x_query = self.layer2(x_query)  # [400, 64, 9, 9] ==> [400, 64, 4, 4]

        x_support_flat = x_support.view(self.n_way * self.n_way * self.n_query, -1)  # [400, 1024]
        x_query_flat = x_query.view(self.n_way * self.n_way * self.n_query, -1)  # [400, 1024]
        cosine = F.cosine_similarity(x_support_flat, x_query_flat, dim=1).view(self.n_way * self.n_way * self.n_query)

        return cosine

class AEAModule(nn.Module):
    def __init__(self, inplanes, scale_value=50, from_value=0.4, value_interval=0.5):
        super(AEAModule, self).__init__()
        self.inplanes = inplanes
        self.scale_value = scale_value
        self.from_value = from_value
        self.value_interval = value_interval
        self.neighbor_k = 15
        self.f_psi = nn.Sequential(
            nn.Linear(self.inplanes, self.inplanes // 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.inplanes // 16, 1),
            nn.Sigmoid()
        )
    def forward(self, x, f_x):

        b, hw, c = x.size()
        clamp_value = self.f_psi(x.view(b * hw, c)) * self.value_interval + self.from_value
        clamp_value = clamp_value.view(b, hw, 1)
        clamp_fx = torch.sigmoid(self.scale_value * (f_x - clamp_value))
        attention_mask = F.normalize(clamp_fx, p=1, dim=-1)


        return attention_mask

class ATLModule(nn.Module):
    def __init__(self, inplanes, transfer_name='W', scale_value=30, atten_scale_value=50, from_value=0.5,
                 value_interval=0.3):
        super(ATLModule, self).__init__()

        self.inplanes = inplanes
        self.scale_value = scale_value
        self.conv = nn.Sequential(
            nn.Conv2d(kernel_size=1,in_channels=5,out_channels=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(kernel_size=1,in_channels=80,out_channels=1)
        self.neighbor_k = 400
        if transfer_name == 'W':
            self.W = nn.Sequential(
                nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.inplanes),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            raise RuntimeError

        self.attention_layer = AEAModule(self.inplanes, atten_scale_value, from_value, value_interval)

    def forward(self, query_data, support_data):
        b, c, h, w = query_data.size() # 80，64，19，19
        s, _, _, _ = support_data.size() # 5，64，19，19
        support_data = support_data.unsqueeze(0).expand(b, -1, -1, -1, -1).contiguous().view(b * s, c, h, w) # 400 ，64，19，19
        w_query = self.W(query_data).view(b, c, h * w)
        w_query = w_query.permute(0, 2, 1).contiguous()
        w_support = self.W(support_data).view(b, s, c, h * w).permute(0, 2, 1, 3).contiguous().view(b, c, s * h * w) # 1*1 conv 80,64,1805
        w_query = F.normalize(w_query, dim=2)
        w_support = F.normalize(w_support, dim=1)

        f_x = torch.matmul(w_query, w_support)

        query_data = query_data.view(b, c, h * w).permute(0, 2, 1) # 80,361,64
        support_data = support_data.view(b, s, c, h * w).permute(0, 2, 1, 3).contiguous().view(b, c, s * h * w) # 80,64,1805
        query_data = F.normalize(query_data, dim=2) # 80,361,64
        support_data = F.normalize(support_data, dim=1) # 80,64,1805

        match_score = torch.matmul(query_data, support_data)
        topk_value1, _ = torch.topk(f_x, self.neighbor_k, 2)
        topk_min1 = torch.min(topk_value1,dim = 2)[0] * 0.4 + 0.4  # 80,361
        topk_min1 = topk_min1.unsqueeze(2) # 80,361,1
        clamp_fx = torch.sigmoid(self.scale_value * (match_score - topk_min1))
        attention_score = F.normalize(clamp_fx, p=1, dim=-1)

        attention_match_score = torch.mul(attention_score, match_score).view(b, h * w, s, h * w).permute(0, 2, 1, 3)


        final_local_score = torch.sum(attention_match_score.contiguous().view(b, s, h * w, h * w), dim=-1) #torch.Size([80, 5, 361])

        final_score = torch.mean(final_local_score, dim=-1) * self.scale_value  # 80 ，5
        final_score = F.normalize(final_score, dim=1)
        return final_score

class OurNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, loss_type='mse'):
        super(OurNet, self).__init__(model_func, n_way, n_support)

        self.loss_type = loss_type  # 'softmax'# 'mse'

        self.classifier_cosine = Classifier_cosine(n_way=self.n_way, n_query=self.n_query, n_support=self.n_support,
                                                   feat_dim=self.feat_dim)

        self.metric_layer = ATLModule(inplanes=64)
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc_se2 = nn.Sequential(
            nn.Linear(64, 64 // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64 // 8, 64, bias=False),
            nn.Sigmoid()
        )
        self.Norm_Layer = nn.BatchNorm1d(self.n_way * 2,affine=True)
        self.FC_Layer = nn.Conv1d(1,1,kernel_size=2,stride=1,dilation=5,bias=False)
        if self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False):
        z_support, z_query, z_support_local,z_query_local= self.parse_feature(x, is_feature)
        z_support_save = z_support.mean(1)
        z_support_save = z_support_save.contiguous().view(self.n_way , self.feat_dim[0], self.feat_dim[1],self.feat_dim[1])
        z_query_save = z_query.contiguous().view(self.n_way * self.n_query, self.feat_dim[0], self.feat_dim[1],self.feat_dim[1])
        cosine = self.classifier_cosine(z_support, z_query).view(-1, self.n_way)  # [80,5]
        local_score = self.metric_layer(z_query_save, z_support_save)
        cosine1 = self.local_cos(z_support_local, z_query_local).view(-1, self.n_way)
        return local_score,cosine,cosine1
    def local_cos(self,z_support,z_query):
        z_support = z_support.mean(1)
        z_support = z_support.contiguous().view(-1,1024)
        z_query = z_query.contiguous().view(-1,1024)
        z_support = z_support.unsqueeze(0).repeat(self.n_query * self.n_way, 1, 1)
        z_query = z_query.unsqueeze(0).repeat(self.n_way, 1, 1)
        z_support = z_support.view(-1, 1024)
        z_query = z_query.contiguous().view(-1, 1024)
        cosine = F.cosine_similarity(z_support, z_query ,dim=1).view(self.n_way*self.n_way*self.n_query)

        return cosine
    def set_forward_loss(self, x):
        y = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        fin_score,cosine, cosine1 = self.set_forward(x)

        if self.loss_type == 'mse':
            y_oh = utils.one_hot(y, self.n_way)
            y_oh = Variable(y_oh.cuda())

            loss1 = self.loss_fn(fin_score, y_oh)
            loss2 = self.loss_fn(cosine, y_oh)
            loss3 = kldiv(cosine1.softmax(dim=-1).log(),cosine.softmax(dim=-1))/80
            loss = loss1  + loss2 + 0.4 * loss3
            return loss

        else:
            y = Variable(y.cuda())
            return self.loss_fn(scores, y)
    def local_att(self, x):

        b,c,_,_ = x.size()
        x1 = self.avg_pool1(x).view(b,c)
        x1 = self.fc_se2(x1).view(b,c,1,1)
        att = x * x1.expand_as(x)

        return att
