import torch.nn as nn
import torch.nn.functional as F
import torch

def conv_bn_relu(in_planes, out_planes, kernel=3, stride=1, padding=1):
     net = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=1),
                         nn.BatchNorm2d(num_features=out_planes),
                         nn.ReLU(True))
     return net

# input size Bx3x224x224
class SegmenterModel(nn.Module):
    def __init__(self, in_size=3, D1=16, D2=32, D3=64):
        super(SegmenterModel, self).__init__()
        self.D1 = D1
        self.D2 = D2
        self.D3 = D3
        
        self.high_res_down = nn.Sequential(
            conv_bn_relu(in_size, D1),
            conv_bn_relu(D1, D1)
        )

        self.med_res_down = nn.Sequential(
            conv_bn_relu(D1, D2),
            conv_bn_relu(D2, D2),
            conv_bn_relu(D2, D2)
        )

        self.low_res = nn.Sequential(
            conv_bn_relu(D2, D3),
            conv_bn_relu(D3, D3),
            conv_bn_relu(D3, D2)
        )

        self.med_res_up = nn.Sequential(
            conv_bn_relu(D2*2, D2),
            conv_bn_relu(D2, D2),
            conv_bn_relu(D2, D1)
        )

        self.high_res_up = nn.Sequential(
            conv_bn_relu(D1*2, D1)
        )

        self.output = nn.Sequential(
            nn.Conv2d(D1, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
         
    def forward(self, input):
        high_res = self.high_res_down(input)
        out, high_indc = F.max_pool2d(high_res, return_indices=True, kernel_size=2, stride=2)

        med_res = self.med_res_down(out)
        out, med_indc = F.max_pool2d(med_res, return_indices=True, kernel_size=2, stride=2)

        out = self.low_res(out)

        out = F.max_unpool2d(out, med_indc, kernel_size=2, stride=2)
        out = torch.cat([med_res, out], 1)
        out = self.med_res_up(out)

        out = F.max_unpool2d(out, high_indc, kernel_size=2, stride=2)
        out = torch.cat([high_res, out], 1)
        out = self.high_res_up(out)

        out = self.output(out)
        return out


class BCECriterion(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BCECriterion, self).__init__()
        self.bce_loss = nn.BCELoss(weight, reduction='mean')

    def forward(self, input, targets):
        probs_flat = input.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)






