
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from config import update_config
from config import config
import argparse
import lib.dataset
import matplotlib.pyplot as plt
import json
import os

from model import hrnet
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description='Train HRNet by jihye')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='./experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml',
                        type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument('--Dataset', default='coco', type=str, dest='Dataset')
    parser.add_argument('--DatasetRoot', default='/media/hjh/2T/data/COCO/', type=str, dest='DatasetRoot')
    parser.add_argument('--DatasetTrainset', default='train2014', type=str, dest='DatasetTrainset')

    args = parser.parse_args()

    return args



class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)


        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()


            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


def train(config, train_loader, model, criterion, optimizer,epoch):

    # switch to train mode
    model.train()


    for i, (input, target, target_weight, meta) in enumerate(train_loader):

        input = torch.autograd.Variable(input).cuda()
        target_weight = torch.autograd.Variable(target_weight).cuda()
        target = torch.autograd.Variable(target).cuda()

        output = model(input)
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        loss = criterion(output, target, target_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if i % config.PRINT_FREQ == 0:
            print( 'epoch: %d, i: %d, loss: %0.5f'% (epoch, i, loss))




def main(args):
    # coco_file = open('/media/hjh/2T/app/human-pose-estimation.pytorch/data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json').read()
    # coco_data = json.loads(coco_file)

    model = hrnet()
    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    print(model)


    ### coco data loader #####
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = eval('lib.'+'dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_SET,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )


    criterion = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()
    optimizer = optim.Adam( model.parameters(),lr=config.TRAIN.LR)
    final_output_dir = './output/'
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        train(config, train_loader, model , criterion, optimizer, epoch)

        if epoch % 10 == 0:
            final_model_state_file = os.path.join(final_output_dir, 'final_state_%d.pth.tar'%epoch)

            torch.save({'state_dict': model.state_dict()}, final_model_state_file)



if __name__ == '__main__':

    args = parse_args()
    main(args)