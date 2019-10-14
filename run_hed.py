import os
from time import time

from HEDNet import HEDNet, ClassBalancedCrossEntropy
import torch
import torch.utils.data as Data
import torch.nn as nn
import pathlib as plb
import scipy.io as scio
import cv2 as cv

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'
MODEL_PATH = 'hed-vgg.pkl'


class ImageDataset(Data.Dataset):
    def __init__(self, img_path: plb.Path, gt_path: plb.Path = None):
        self.img_path = img_path
        self.gt_path = gt_path
        self.ls = []
        for jpg in self.img_path.glob('*.jpg'):
            if self.gt_path is not None:
                gt = self.gt_path.glob(f'{jpg.stem}.mat')
                if len(list(gt)) > 0:
                    self.ls.append(jpg.stem)
            else:
                self.ls.append(jpg.stem)

    def __getitem__(self, index):
        fname = self.ls[index]
        imgarr = cv.imread(f'{self.img_path}/{fname}.jpg', cv.IMREAD_COLOR)
        if imgarr.shape[0] > 500:
            imgarr = cv.resize(src=imgarr, dsize=(480, 480))
        b, g, r = cv.split(imgarr)
        imgarr = cv.merge([r, g, b])
        imgarr = torch.tensor(imgarr, dtype=torch.float)
        imgarr = imgarr.permute(2, 0, 1)
        if self.gt_path is None:
            return imgarr

        mat = scio.loadmat(f'{self.gt_path}/{fname}.mat')
        # groundTruth = []
        # for i in range(6):
        #     gt = mat['groundTruth']
        #     groundTruth.append(torch.tensor(gt[0][i][0][0][1], dtype=torch.float))
        groundTruth = torch.tensor(mat['groundTruth'][0][0][0][0][1], dtype=torch.float)
        groundTruth = torch.unsqueeze(groundTruth, dim=0)
        if imgarr.shape[1] == 481:
            # print(imgarr.shape,groundTruth.shape)
            imgarr = imgarr.permute(0, 2, 1)
            groundTruth = groundTruth.permute(0, 2, 1)
        return imgarr, groundTruth

    def __len__(self):
        return len(self.ls)


def calc_acc(input: torch.Tensor, target: torch.Tensor):
    input[input >= 0.7] = 1
    input[input < 0.7] = 0
    input = input.cpu().int()
    target = target.cpu().int()
    tot = target.sum().item()
    P = input.sum().item()
    cnt = (input.__and__(target)).sum().item()
    R = cnt / tot
    P = cnt / (P + 0.001)
    F1 = 2 * P * R / (P + R + 0.001)
    return F1


def makejpg(imgarr: torch.Tensor, filename):
    imgarr = imgarr.detach()
    imgarr = imgarr * 255
    img = imgarr.cpu().numpy()
    # beta = img.max() / 255
    # img = img / beta
    # img[img > 0.8] = 255
    # img[img <= 0.8] = 0
    cv.imwrite(filename, img)


def validate(net, dataset, BATCH_SIZE=5):
    t1 = time()
    net.eval()
    dataloader = Data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)
    loss_fn = ClassBalancedCrossEntropy().cuda()
    v_loss = []
    v_acc = []
    cnt = 0
    for bi, (bx, by) in enumerate(dataloader):
        bx = bx.cuda()
        by = by.cuda()
        output = net(bx)
        # losses = sum([loss_fn(y_hat, by) for y_hat in output])
        # losses = losses / 6
        # v_loss.append(losses.item())
        # acc = calc_acc(output, by)
        # v_acc.append(acc)
        for i in range(output[-1].shape[0]):
            cnt += 1
            makejpg(output[-1][i][0], f'val/val-{cnt}.jpg')
    # print('validate: time', time() - t1, ' loss', sum(v_loss) / len(v_loss))


def train(net, train_dataset, validate_dataset,
          LR=5e-4, EPOCH=120, BATCH_SIZE=40, use_gpu=True):
    loss_fn = ClassBalancedCrossEntropy()
    if use_gpu: loss_fn = loss_fn.cuda()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    optsch = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=2, gamma=0.95)
    dataloader = Data.DataLoader(dataset=train_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=4)
    for epoch in range(EPOCH):
        net.train()
        print('epoch:', epoch + 1)
        t1 = time()
        t_loss=[]
        for bi, (bx, by) in enumerate(dataloader):
            print('step', bi + 1, end='\t')
            opt.zero_grad()
            if use_gpu:
                bx = bx.cuda()
                by = by.cuda()
            output = net(bx)
            alpha = [0.05, 0.1, 0.15, 0.2, 0.2, 0.3]
            losses = sum([loss_fn(y[0], by, y[1]) for y in list(zip(output, alpha))])
            t_loss.append(losses.item())
            losses.backward()
            opt.step()
            # if (bi + 1) % 5 == 0:
            print('loss', losses.item())
        optsch.step()
        print(sum(t_loss)/len(t_loss))
        net.eval()
        print('time', time() - t1)
        if (epoch + 1) % 10 == 0:
            validate(net, validate_dataset)
            dic = net.state_dict()
            dic2 = {}
            if use_gpu:
                keys = dic.keys()
                for k in keys:
                    dic2[k.split('.', 1)[-1]] = dic[k]
            torch.save(dic2, 'hed-vgg.pkl')
            print('model saved')
        print('__________________________________________________')


def test(net, dataset, batch_size=10):
    net.eval()
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, num_workers=8)
    cnt = 0
    if dataset.gt_path is not None:
        acc = []
        for bx, by in dataloader:
            bx = bx.cuda()
            by = by.cuda()
            output = net(bx)
            # f1 = calc_acc(output, by)
            # acc.append(f1)
            for i in range(batch_size):
                cnt += 1
                makejpg(output[-1][i][0], f'pic/test{cnt}.jpg')
        # assert len(acc) > 0
        # print(sum(acc) / len(acc))
    else:
        for bx in dataloader:
            bx = bx.cuda()
            output = net(bx)
            output = sum(output) / 6
            for i in range(batch_size):
                cnt += 1
                makejpg(output[i][0], f'pic/test{cnt}.jpg')


DATA_DIR = plb.Path('BSR/BSDS500/data')
IMAGE_DIR = DATA_DIR.joinpath('images/')
GT_DIR = DATA_DIR.joinpath('groundTruth')
TRAIN_GT = GT_DIR.joinpath('train')
TEST_GT = GT_DIR.joinpath('test')
VALID_GT = GT_DIR.joinpath('val')
TRAIN_DIR = IMAGE_DIR.joinpath('train')
VALID_DIR = IMAGE_DIR.joinpath('val')
TEST_DIR = IMAGE_DIR.joinpath('test')


def main():
    net = HEDNet()

    parameters = torch.load('hed-vgg.pkl')
    print(parameters.keys())
    net.load_state_dict(parameters)

    def load_from_vgg16():
        d1 = torch.load('models/vgg16-bn.pth')
        k1 = list(d1.keys())
        d2 = net.state_dict()
        k2 = list(d2.keys())
        j = 0
        for i in range(78):
            while k2[j].split('.')[-1] != k1[i].split('.')[-1]:
                j += 1
            d2[k2[j]] = d1[k1[i]]
        net.load_state_dict(d2, strict=False)

    # load_from_vgg16()
    net = net.cuda()
    net = nn.DataParallel(net)

    train_dataset = ImageDataset(TRAIN_DIR, TRAIN_GT)
    validate_dataset = ImageDataset(VALID_DIR, VALID_GT)
    test_dataset = ImageDataset(TEST_DIR, TEST_GT)
    train(net, train_dataset, validate_dataset)
    # test(net, test_dataset)


if __name__ == '__main__':
    main()
