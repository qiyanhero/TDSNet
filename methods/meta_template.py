import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import utils
from abc import abstractmethod

from methods.triplet_loss import construct_triplets, TripleLoss

cross_entropy_loss = nn.CrossEntropyLoss()

class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1  # (change depends on input)
        self.feature = model_func()
        self.feat_dim = self.feature.final_feat_dim
        self.inpalnes = 64
        self.change_way = change_way  # some methods allow different_way classification during training and test
        #self.fc = nn.Linear(16 * 64, 200, bias=False)

    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        x = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all,z_local = self.feature.forward(x)

            z_all = z_all.view(self.n_way, self.n_support + self.n_query, self.feat_dim[0], self.feat_dim[1],
                               self.feat_dim[1])
            z_local1 = z_local.view(self.n_way, self.n_support + self.n_query, -1)  # [5，17，1024]

        z_support_local = z_local1[:, :self.n_support]  # [5,1,1024]
        z_query_local = z_local1[:, self.n_support:]  # [5,16,1024]
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]
        return z_support, z_query,z_support_local,z_query_local

    def correct(self, x):

        if self.__class__.__name__ == "OurNet":
            score1, score2, _  = self.set_forward(x)
            scores = (score1 + score2) / 2.


        else:
            scores,_ = self.set_forward(x)

        y_query = np.repeat(range(self.n_way), self.n_query)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)

        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)

        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer, **kwargs):
        print_freq = 10
        beta = 5e-2
        tripleloss = TripleLoss()
        avg_loss = 0
        center_loss = utils.CenterLoss()
        for i, (x, label) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            optimizer.zero_grad()
            loss1= self.set_forward_loss(x)
            loss = loss1
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))

    def test_loop(self, test_loader, record=None):
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return acc_mean


