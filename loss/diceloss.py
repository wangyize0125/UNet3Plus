#! python3
# encoding: utf-8
"""
@author:    Yize Wang
@contact:    wangyize@hust.edu.cn
@file:    diceloss.py
@time:   2022/3/16 18:28
@description:    
"""

import torch
import torch.nn as nn

"""
Dice loss: Dice系数是一种集合相似度的度量函数，通常用于计算两个样本之间的相似度，取值范围在[0，1]之间，公式如下：
                                s = 2 * |X^Y| / (|X| + |Y|)
           其中|X^Y|是X和Y之间的交集，|X|和|Y|分别表示X和Y的元素的个数，其中，分子的系数是2，是因为分母存在重复计算X和Y之间的共同元素
           
           Ldice可以表示为1 -（2I + epsilon）/（U + epsilon），I为真实和预测（sigmoid和softmax处理之后的结果）结果乘积之和，U为
                真实值的平方和预测值的平方之和
           Ldice也可以表示为1 -（I + epsilon）/（U - I + epsilon），I为真实和预测（sigmoid和softmax处理之后的结果）结果乘积之和，U为
                真实值的平方和预测值的平方之和
"""


class DiceLoss(nn.Module):
    """ dice coeff for individual examples """

    epsilon = 0.0000001

    def __init__(self, input_data, target_data):
        super(DiceLoss, self).__init__()

        self.input_data = input_data
        self.target_data = target_data

        assert input_data.size() == target_data.size(), "Input and target data have different sizes: {} and {}".format(
            input_data.size(), target_data.size()
        )

    def forward(self):
        if self.input_data.is_cuda:
            loss = torch.FloatTensor(1).cuda().zero_()
        else:
            loss = torch.FloatTensor(1).zero_()

        # calculate each figure
        for com in zip(self.input_data, self.target_data):
            inter = torch.dot(com[0].view(-1), com[1].view(-1))
            union = torch.sum(com[0]) + torch.sum(com[1])

            loss += 1 - (2 * inter + DiceLoss.epsilon) / (union + DiceLoss.epsilon)


        loss /= self.input_data.size()[0]

        return loss


class FocalLoss(nn.Module):
    def __init__(self, preds, labels, alpha=0.25, gamma=2, num_classes=1, size_average=True):
        super(FocalLoss, self).__init__()

        self.size_average = size_average
        self.preds = preds
        self.labels = labels

        if isinstance(alpha, list):
            assert len(alpha) == num_classes, "{} class but {} alpha".format(num_classes, len(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            # self.alpha[1:] += (1 - alpha)

        self.gamma = gamma

    def forward(self):
        # assert preds.dim()==2 and labels.dim()==1
        preds = self.preds.view(-1, 1)
        labels = self.labels.view(-1, 1)
        self.alpha = self.alpha.to(preds.device)

        # focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        pt1 = torch.where(torch.eq(labels, 1), preds, torch.ones_like(preds))
        pt0 = torch.where(torch.eq(labels, 0), preds, torch.zeros_like(preds))
        pt1 = torch.clip(pt1, 0.0001, 0.9999)
        pt0 = torch.clip(pt0, 0.0001, 0.9999)

        loss = -torch.mean(self.alpha * torch.pow(1 - pt1, self.gamma) * torch.log(pt1)) - torch.mean((1 - self.alpha) * torch.pow(pt0, self.gamma) * torch.log(1 - pt0))

        return loss


