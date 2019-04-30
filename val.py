
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from config import update_config
from config import config
import argparse
import matplotlib.pyplot as plt
import json
import os
import cv2
import numpy as np
import dataset
import math

from model import hrnet


def parse_args():
    parser = argparse.ArgumentParser(description='Train HRNet by jihye')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='./experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml',
                        type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    args = parser.parse_args()

    return args


def get_kpts(maps, img_h = 368.0, img_w = 368.0):

    # maps (1,15,46,46)
    # maps = maps.clone().cpu().data.numpy()
    # map_6 = maps[0]

    map_6 = maps

    kpts = []
    for i in range(17):
    # for m in map_6[1:]:
        m = maps[:,:,i]
        h, w = np.unravel_index(m.argmax(), m.shape)
        x = int(w * img_w / m.shape[1])
        y = int(h * img_h / m.shape[0])
        kpts.append([x,y])
    return kpts

def draw_paint(im, kpts):
    im = im.cpu().data.numpy().transpose((1, 2, 0))
    fig, ax = plt.subplots()

    for k in kpts:
        x = int(k[0])
        y = int(k[1])

        circle = plt.Circle((x, y), 2, color='r')
        ax.add_artist(circle)
        plt.imshow(im)
        plt.show()

        #
        # cv2.circle(im, (x, y), radius=1, thickness=-1, color=(0, 0, 255))
        # plt.imshow(im)
        # plt.show()
        # # cv2.imshow('test_example', im)
        # # cv2.waitKey(10)


def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped



def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array([hm[py][px+1] - hm[py][px-1],
                                     hm[py+1][px]-hm[py-1][px]])
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize#[n]
                normed_targets = target[n, c, :] / normalize#[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred

def main():
    coco_file = open('/media/hjh/2T/app/human-pose-estimation.pytorch/data/coco/annotations/person_keypoints_val2017.json').read()
    coco_data = json.loads(coco_file)

    dict_out = {'annotations': [], 'categories': [], 'images': []}
    dict_out['categories'] = coco_data['categories']
    dict_out['images'] = coco_data['images']

    model = hrnet()
    dir_pth = './output/final_state_70.pth.tar'
    state_dict = torch.load(dir_pth)['state_dict']
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    model.eval()

    ### coco data loader #####
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    num_samples = len(valid_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    idx = 0
    image_path = []
    filenames = []
    imgnums = []

    for i, (input, target, target_weight, meta) in enumerate(valid_loader):
        # input = torch.autograd.Variable(input).cuda()
        print((i*config.TEST.BATCH_SIZE * len(gpus)), num_samples)

        output = model(input)

        # draw_paint(input[0], kpts)

        if config.TEST.FLIP_TEST:
            input_flipped = np.flip(input.cpu().numpy(), 3).copy()
            input_flipped = torch.from_numpy(input_flipped).cuda()
            output_flipped = model(input_flipped)
            output_flipped = flip_back(output_flipped.cpu().numpy(),
                                       valid_dataset.flip_pairs)
            output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

            if config.TEST.SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = \
                    output_flipped.clone()[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5


        num_images = input.size(0)

        c = meta['center'].numpy()
        s = meta['scale'].numpy()
        score = meta['score'].numpy()

        preds, maxvals = get_final_preds(
            config, output.clone().detach().cpu().numpy(), c, s)

        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        # double check this all_boxes parts
        all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
        all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
        all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
        all_boxes[idx:idx + num_images, 5] = score
        image_path.extend(meta['image'])
        if config.DATASET.DATASET == 'posetrack':
            filenames.extend(meta['filename'])
            imgnums.extend(meta['imgnum'].numpy())


        idx += num_images



        for b in range(len(meta['joints'])):
            reshape_pred = np.concatenate((preds[b], np.ones((17, 1))), axis=1).reshape(-1)

            reshape_pred = reshape_pred.tolist()
            anno_dict = {
                         'area': float((meta['bbox'][2][b] * meta['bbox'][3][b]).data.numpy()),
                         # 'bbox': [float(meta['bbox'][0][b].data.numpy()), float(meta['bbox'][1][b].data.numpy()),
                         #          float(meta['bbox'][2][b].data.numpy()), float(meta['bbox'][3][b].data.numpy())],
                         'category_id': 1,
                         'id': int(meta['id'][b].data.numpy()),
                         'image_id': int(meta['image_id'][b].data.numpy()),
                         'iscrowd': 0,
                         'keypoints': reshape_pred,
                         'num_keypoints': len(meta['joints'][b].data.numpy()),
                         'score': float(score[b])

                         }



            dict_out['annotations'].append(anno_dict)



    ### save json
    with open('all_coco_val_pred.json', 'w') as fp:
        json.dump(dict_out, fp)






if __name__ == '__main__':

    args = parse_args()

    main()


