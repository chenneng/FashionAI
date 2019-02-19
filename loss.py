import torch
import torch.nn as nn
import torch.nn.functional as F

class fashionLoss(nn.Module):
    def __init__(self):
        super(fashionLoss, self).__init__()

    def forward(self, outputN, label):
        loss = 0

        output_attr1 = outputN[0]
        output_attr2 = outputN[1]
        output_attr3 = outputN[2]
        output_attr4 = outputN[3]

        for i, output in enumerate(output_attr1):

            label1 = label[i][0:8]
            label2 = label[i][8:14]
            label3 = label[i][14:20]
            label4 = label[i][20:29]

            softmax1 = F.softmax(output_attr1[i], dim = 0)
            softmax2 = F.softmax(output_attr2[i], dim = 0)
            softmax3 = F.softmax(output_attr3[i], dim = 0)
            softmax4 = F.softmax(output_attr4[i], dim = 0)

            loss1 = -1.0 * torch.sum(torch.mul(torch.log(softmax1 + 1e-10), label1))
            loss2 = -1.0 * torch.sum(torch.mul(torch.log(softmax2 + 1e-10), label2))
            loss3 = -1.0 * torch.sum(torch.mul(torch.log(softmax3 + 1e-10), label3))
            loss4 = -1.0 * torch.sum(torch.mul(torch.log(softmax4 + 1e-10), label4))
            loss += loss1 + loss2 + loss3 +loss4

        size = list(output_attr1.size())
        return loss/size[0]
