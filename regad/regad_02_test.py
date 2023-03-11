# 1. Run Python File
# 2. paste:
#    export LD_LIBRARY_PATH=/home/kswada/kw/mvtech_ad/regad/venv/lib/python3.8/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

import os

import random
import argparse
import time
import math
import numpy as np

from collections import OrderedDict

from tqdm import tqdm

import torch
# import torch.optim as optim
from torch.optim import optimizer
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter

# ----------
from src.models.siamese import Encoder, Predictor
from src.models.stn import stn_net

from src.datasets.mvtec import FSAD_Dataset_train, FSAD_Dataset_test
from src.losses.norm_loss import CosLoss

from src.utils.utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
from src.utils.funcs import embedding_concat, mahalanobis_torch, rot_img, translation_img, hflip_img, rot90_img, grey_img


# ----------
# REFERENCE:
# https://github.com/MediaBrain-SJTU/RegAD
# 2022_Registration Based Few-Shot Anomaly Detection


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# test
# ----------------------------------------------------------------------------------------------------------------

def test(args, models, cur_epoch, fixed_fewshot_list, test_loader, **kwargs):

    # -----------
    STN = models[0]
    ENC = models[1]
    PRED = models[2]

    STN.eval()
    ENC.eval()
    PRED.eval()

    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    support_img = fixed_fewshot_list[cur_epoch]

    ############################################
    augment_support_img = support_img

    # -----------
    # rotate img with small angle
    for angle in [-np.pi/4, -3 * np.pi/16, -np.pi/8, -np.pi/16, np.pi/16, np.pi/8, 3 * np.pi/16, np.pi/4]:
        rotate_img = rot_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate_img], dim=0)

    # -----------
    # translate img
    for a,b in [(0.2,0.2), (-0.2,0.2), (-0.2,-0.2), (0.2,-0.2), (0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
        trans_img = translation_img(support_img, a, b)
        augment_support_img = torch.cat([augment_support_img, trans_img], dim=0)

    # -----------
    # hflip img
    flipped_img = hflip_img(support_img)
    augment_support_img = torch.cat([augment_support_img, flipped_img], dim=0)

    # -----------
    # rgb to grey img
    greyed_img = grey_img(support_img)
    augment_support_img = torch.cat([augment_support_img, greyed_img], dim=0)

    # -----------
    # rotate img in 90 degree
    for angle in [1,2,3]:
        rotate90_img = rot90_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate90_img], dim=0)
    augment_support_img = augment_support_img[torch.randperm(augment_support_img.size(0))]
    ############################################

    # -----------
    # torch version
    with torch.no_grad():
        support_feat = STN(augment_support_img.to(device))
    support_feat = torch.mean(support_feat, dim=0, keepdim=True)
    train_outputs['layer1'].append(STN.stn1_output)
    train_outputs['layer2'].append(STN.stn2_output)
    train_outputs['layer3'].append(STN.stn3_output)

    for k, v in train_outputs.items():
        train_outputs[k] = torch.cat(v, 0)

    # -----------
    # Embedding concat
    embedding_vectors = train_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name], True)

    # -----------
    # calculate multivariate Gaussian distribution
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    mean = torch.mean(embedding_vectors, dim=0)
    cov = torch.zeros(C, C, H * W).to(device)
    I = torch.eye(C).to(device)
    for i in range(H * W):
        cov[:, :, i] = torch.cov(embedding_vectors[:, :, i].T) + 0.01 * I
    train_outputs = [mean, cov]

    # -----------
    # torch version
    query_imgs = []
    gt_list = []
    mask_list = []
    score_map_list = []

    for (query_img, _, mask, y) in tqdm(test_loader):
        query_imgs.extend(query_img.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        mask_list.extend(mask.cpu().detach().numpy())
        
        # model prediction
        query_feat = STN(query_img.to(device))
        z1 = ENC(query_feat)
        z2 = ENC(support_feat)
        p1 = PRED(z1)
        p2 = PRED(z2)

        loss = CosLoss(p1,z2, Mean=False)/2 + CosLoss(p2,z1, Mean=False)/2
        loss_reshape = F.interpolate(loss.unsqueeze(1), size=query_img.size(2), mode='bilinear',align_corners=False).squeeze(0)
        score_map_list.append(loss_reshape.cpu().detach().numpy())

        test_outputs['layer1'].append(STN.stn1_output)
        test_outputs['layer2'].append(STN.stn2_output)
        test_outputs['layer3'].append(STN.stn3_output)

    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)

    # Embedding concat
    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name], True)

    # calculate distance matrix
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    dist_list = []

    for i in range(H * W):
        mean = train_outputs[0][:, i]
        conv_inv = torch.linalg.inv(train_outputs[1][:, :, i])
        dist = [mahalanobis_torch(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)

    dist_list = torch.tensor(dist_list).transpose(1, 0).reshape(B, H, W)

    # upsample
    score_map = F.interpolate(dist_list.unsqueeze(1), size=query_img.size(2), mode='bilinear',
                              align_corners=False).squeeze().numpy()

    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    return score_map, query_imgs, gt_list, mask_list


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# base setting 
# ----------------------------------------------------------------------------------------------------------------

import warnings

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')


# ----------------------------------------------------------------------------------------------------------------
# set argments
# ----------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='RegAD on MVtec')
parser.add_argument('--obj', type=str, default='hazelnut')
parser.add_argument('--data_type', type=str, default='mvtec')
parser.add_argument('--data_path', type=str, default='./MVTec/')
parser.add_argument('--epochs', type=int, default=50, help='maximum training epochs')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=0.01, help='learning rate in SGD')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
parser.add_argument('--seed', type=int, default=668, help='manual seed')
parser.add_argument('--shot', type=int, default=2, help='shot count')
parser.add_argument('--inferences', type=int, default=10, help='number of rounds per inference')
parser.add_argument('--stn_mode', type=str, default='rotation_scale', help='[affine, translation, rotation, scale, shear, rotation_scale, translation_scale, rotation_translation, rotation_translation_scale]')

args = parser.parse_args()

args.input_channel = 3


# ----------
args.obj = 'bottle'
args.data_type = 'mvtec'
args.data_path = '/media/kswada/MyFiles/dataset/mvtec_ad/'
args.epochs = 50
args.batch_size = 32
args.img_size = 224
args.lr = 0.01
args.momentum = 0.9
args.shot = 2
args.inferences = 10
args.stn_mode = 'rotation_scale'


# ----------
args.seed = 668

if args.seed is None:
    args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

if use_cuda:
    torch.cuda.manual_seed_all(args.seed)


# ----------
args.prefix = time_file_str()


# ----------------------------------------------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------------------------------------------

STN = stn_net(args).to(device)

ENC = Encoder().to(device)

PRED = Predictor().to(device)


models = [STN, ENC, PRED]


# ----------------------------------------------------------------------------------------------------------------
# checkpoint
# ----------------------------------------------------------------------------------------------------------------

# provided from author
# CKPT_name = f'./01_data/save_checkpoints/{args.shot}/{args.obj}/{args.obj}_{args.shot}_rotation_scale_model.pt'

# kw train
CKPT_name = f'./04_output/logs_mvtec/rotation_scale/{args.shot}/{args.obj}/{args.obj}_{args.shot}_rotation_scale_model.pt'

print(CKPT_name)


# ----------
model_CKPT = torch.load(CKPT_name)

STN.load_state_dict(model_CKPT['STN'])

ENC.load_state_dict(model_CKPT['ENC'])

PRED.load_state_dict(model_CKPT['PRED'])


# ----------------------------------------------------------------------------------------------------------------
# load dataset
# ----------------------------------------------------------------------------------------------------------------

kwargs = {'num_workers': 12, 'pin_memory': True} if use_cuda else {}

test_dataset = FSAD_Dataset_test(args.data_path, class_name=args.obj, is_train=False, resize=args.img_size, shot=args.shot)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)


# ----------
# test dataset
one_sample = next(iter(test_dataset))

# query image
print(one_sample[0].shape)

# support image:  2 shot
print(one_sample[1][0].shape)
print(one_sample[1][1].shape)

# mask
print(one_sample[2].shape)

# y
print(one_sample[3])


# ----------
# load fixed support set
fixed_fewshot_list = torch.load(f'./01_data/support_set/{args.obj}/{args.shot}_{args.inferences}.pt')

# 2 shot * 10 inferences
print(f'len(fixed_fewshot_list): {len(fixed_fewshot_list)}')
print(fixed_fewshot_list[0].shape)


# ----------------------------------------------------------------------------------------------------------------
# start test
# ----------------------------------------------------------------------------------------------------------------

image_auc_list = []

pixel_auc_list = []


for inference_round in range(args.inferences):

    print('Round {}:'.format(inference_round))
    scores_list, test_imgs, gt_list, gt_mask_list = test(args, models, inference_round, fixed_fewshot_list, test_loader, **kwargs)
    scores = np.asarray(scores_list)

    # ----------
    # Normalization
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    # ----------
    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    image_auc_list.append(img_roc_auc)

    # ----------
    # calculate per-pixel level ROCAUC
    gt_mask = np.asarray(gt_mask_list)
    gt_mask = (gt_mask > 0.5).astype(np.int_)
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    pixel_auc_list.append(per_pixel_rocauc)


# ----------------------------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------------------------

image_auc_list = np.array(image_auc_list)

pixel_auc_list = np.array(pixel_auc_list)

mean_img_auc = np.mean(image_auc_list, axis = 0)

mean_pixel_auc = np.mean(pixel_auc_list, axis = 0)


# ----------
print('Img-level AUC:',mean_img_auc)
print('Pixel-level AUC:', mean_pixel_auc)



##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# check step by step
# ----------------------------------------------------------------------------------------------------------------

from PIL import Image


# ----------
STN = models[0]
ENC = models[1]
PRED = models[2]

STN.eval()
ENC.eval()
PRED.eval()


# ----------
train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])


# ----------
# check support images
for cur_epoch in range(args.inferences):
    support_img = fixed_fewshot_list[cur_epoch]

    for i in range(len(support_img)):
        tmp_img = support_img[i].numpy().astype('uint8')[0] * 255
        image = Image.fromarray(tmp_img)
        image.show()


# ----------
cur_epoch = 5
support_img = fixed_fewshot_list[cur_epoch]


############################################
augment_support_img = support_img

# -----------
# rotate img with small angle
for angle in [-np.pi/4, -3 * np.pi/16, -np.pi/8, -np.pi/16, np.pi/16, np.pi/8, 3 * np.pi/16, np.pi/4]:
    rotate_img = rot_img(support_img, angle)
    augment_support_img = torch.cat([augment_support_img, rotate_img], dim=0)

# -----------
# translate img
for a,b in [(0.2,0.2), (-0.2,0.2), (-0.2,-0.2), (0.2,-0.2), (0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
    trans_img = translation_img(support_img, a, b)
    augment_support_img = torch.cat([augment_support_img, trans_img], dim=0)

# -----------
# hflip img
flipped_img = hflip_img(support_img)
augment_support_img = torch.cat([augment_support_img, flipped_img], dim=0)

# -----------
# rgb to grey img
greyed_img = grey_img(support_img)
augment_support_img = torch.cat([augment_support_img, greyed_img], dim=0)

# -----------
# rotate img in 90 degree
for angle in [1,2,3]:
    rotate90_img = rot90_img(support_img, angle)
    augment_support_img = torch.cat([augment_support_img, rotate90_img], dim=0)
augment_support_img = augment_support_img[torch.randperm(augment_support_img.size(0))]
############################################


# -----------
# torch version
with torch.no_grad():
    support_feat = STN(augment_support_img.to(device))

support_feat = torch.mean(support_feat, dim=0, keepdim=True)

train_outputs['layer1'].append(STN.stn1_output)
train_outputs['layer2'].append(STN.stn2_output)
train_outputs['layer3'].append(STN.stn3_output)

for k, v in train_outputs.items():
    train_outputs[k] = torch.cat(v, 0)

print(train_outputs['layer1'])


# -----------
# Embedding concat
embedding_vectors = train_outputs['layer1']

for layer_name in ['layer2', 'layer3']:
    embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name], True)

print(embedding_vectors.shape)


# -----------
# calculate multivariate Gaussian distribution
B, C, H, W = embedding_vectors.size()
embedding_vectors = embedding_vectors.view(B, C, H * W)
mean = torch.mean(embedding_vectors, dim=0)
cov = torch.zeros(C, C, H * W).to(device)
I = torch.eye(C).to(device)
for i in range(H * W):
    cov[:, :, i] = torch.cov(embedding_vectors[:, :, i].T) + 0.01 * I
train_outputs = [mean, cov]

# -----------
# torch version
query_imgs = []
gt_list = []
mask_list = []
score_map_list = []

for (query_img, _, mask, y) in tqdm(test_loader):
    query_imgs.extend(query_img.cpu().detach().numpy())
    gt_list.extend(y.cpu().detach().numpy())
    mask_list.extend(mask.cpu().detach().numpy())
    
    # model prediction
    query_feat = STN(query_img.to(device))
    z1 = ENC(query_feat)
    z2 = ENC(support_feat)
    p1 = PRED(z1)
    p2 = PRED(z2)

    loss = CosLoss(p1,z2, Mean=False)/2 + CosLoss(p2,z1, Mean=False)/2
    loss_reshape = F.interpolate(loss.unsqueeze(1), size=query_img.size(2), mode='bilinear',align_corners=False).squeeze(0)
    score_map_list.append(loss_reshape.cpu().detach().numpy())

    test_outputs['layer1'].append(STN.stn1_output)
    test_outputs['layer2'].append(STN.stn2_output)
    test_outputs['layer3'].append(STN.stn3_output)

for k, v in test_outputs.items():
    test_outputs[k] = torch.cat(v, 0)

# Embedding concat
embedding_vectors = test_outputs['layer1']
for layer_name in ['layer2', 'layer3']:
    embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name], True)

# calculate distance matrix
B, C, H, W = embedding_vectors.size()
embedding_vectors = embedding_vectors.view(B, C, H * W)
dist_list = []

for i in range(H * W):
    mean = train_outputs[0][:, i]
    conv_inv = torch.linalg.inv(train_outputs[1][:, :, i])
    dist = [mahalanobis_torch(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
    dist_list.append(dist)

dist_list = torch.tensor(dist_list).transpose(1, 0).reshape(B, H, W)

# upsample
score_map = F.interpolate(dist_list.unsqueeze(1), size=query_img.size(2), mode='bilinear',
                            align_corners=False).squeeze().numpy()

# apply gaussian smoothing on the score map
for i in range(score_map.shape[0]):
    score_map[i] = gaussian_filter(score_map[i], sigma=4)


# ----------
# scores_list, test_imgs, gt_list, gt_mask_list = test(args, models, inference_round, fixed_fewshot_list, test_loader, **kwargs)

scores = np.asarray(scores_list)


# ----------
# Normalization
max_anomaly_score = scores.max()
min_anomaly_score = scores.min()
scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

# ----------
# calculate image-level ROC AUC score
img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
gt_list = np.asarray(gt_list)
img_roc_auc = roc_auc_score(gt_list, img_scores)
image_auc_list.append(img_roc_auc)

# ----------
# calculate per-pixel level ROCAUC
gt_mask = np.asarray(gt_mask_list)
gt_mask = (gt_mask > 0.5).astype(np.int_)
per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
pixel_auc_list.append(per_pixel_rocauc)


