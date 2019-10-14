import torch
import torch.nn as nn
import torch.nn.functional as F


class HEDNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)

        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)

        self.relu = nn.ReLU()

        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn5 = nn.Conv2d(512, 1, 1)
        self.score_final = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        image_h, image_w = x.shape[2], x.shape[3]

        con1 = self.relu(self.bn1_1(self.conv1_1(x)))
        con1 = self.relu(self.bn1_2(self.conv1_2(con1)))
        pool_1 = self.maxpool(con1)

        con2 = self.relu(self.bn2_1(self.conv2_1(pool_1)))
        con2 = self.relu(self.bn2_2(self.conv2_2(con2)))
        pool_2 = self.maxpool(con2)

        con3 = self.relu(self.bn3_1(self.conv3_1(pool_2)))
        con3 = self.relu(self.bn3_2(self.conv3_2(con3)))
        con3 = self.relu(self.bn3_3(self.conv3_3(con3)))
        pool_3 = self.maxpool(con3)

        con4 = self.relu(self.bn4_1(self.conv4_1(pool_3)))
        con4 = self.relu(self.bn4_2(self.conv4_2(con4)))
        con4 = self.relu(self.bn4_3(self.conv4_3(con4)))
        pool_4 = self.maxpool(con4)

        con5 = self.relu(self.bn5_1(self.conv5_1(pool_4)))
        con5 = self.relu(self.bn5_2(self.conv5_2(con5)))
        con5 = self.relu(self.bn5_3(self.conv5_3(con5)))

        dsn1 = self.dsn1(con1)
        dsn2 = self.dsn2(con2)
        dsn3 = self.dsn3(con3)
        dsn4 = self.dsn4(con4)
        dsn5 = self.dsn5(con5)

        upsample1 = dsn1
        upsample2 = F.interpolate(dsn2, (image_h, image_w), mode='bilinear')
        upsample3 = F.interpolate(dsn3, (image_h, image_w), mode='bilinear')
        upsample4 = F.interpolate(dsn4, (image_h, image_w), mode='bilinear')
        upsample5 = F.interpolate(dsn5, (image_h, image_w), mode='bilinear')

        fuse_cat = torch.cat((upsample1, upsample2, upsample3, upsample4, upsample5), dim=1)
        fuse = self.score_final(fuse_cat)  # Shape: [batch_size, 1, image_h, image_w].
        results = [torch.sigmoid(upsample1),
                   torch.sigmoid(upsample2),
                   torch.sigmoid(upsample3),
                   torch.sigmoid(upsample4),
                   torch.sigmoid(upsample5),
                   torch.sigmoid(fuse)]
        return results


class ClassBalancedCrossEntropy(nn.Module):
    def forward(self, input: torch.Tensor, target: torch.Tensor, weight=1):
        # since pos is less than neg, the loss at (Y=1) has a higher
        # weight to balance the result
        pos = target.sum()
        b, c, h, w = target.shape
        tot = b * c * h * w
        pos = pos / tot
        neg = 1 - pos
        beta = torch.zeros_like(target)
        beta[target >= 0.5] = neg
        beta[target < 0.5] = pos
        beta = beta * weight
        return F.binary_cross_entropy(input, target, beta)
