
import os
from glob import glob

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm

# from PIL import Image
import PIL

import yaml
from ignite.contrib import metrics

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import numpy as np
import tqdm

import matplotlib.pyplot as plt


########################################################################################################################
# -----------------------------------------------------------------------------------------------------------------------
# constants
# -----------------------------------------------------------------------------------------------------------------------

_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

BACKBONE_DEIT = "deit_base_distilled_patch16_384"
BACKBONE_CAIT = "cait_m48_448"
BACKBONE_RESNET18 = "resnet18"
BACKBONE_WIDE_RESNET50 = "wide_resnet50_2"

SUPPORTED_BACKBONES = [
    BACKBONE_DEIT,
    BACKBONE_CAIT,
    BACKBONE_RESNET18,
    BACKBONE_WIDE_RESNET50,
]


########################################################################################################################
# -----------------------------------------------------------------------------------------------------------------------
# dataset
# -----------------------------------------------------------------------------------------------------------------------

from enum import Enum

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MVTecDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate


########################################################################################################################
# -----------------------------------------------------------------------------------------------------------------------
# utils
# -----------------------------------------------------------------------------------------------------------------------

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


########################################################################################################################
# -----------------------------------------------------------------------------------------------------------------------
# fastflow
# -----------------------------------------------------------------------------------------------------------------------

def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv


def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class FastFlow(nn.Module):
    def __init__(
        self,
        backbone_name,
        flow_steps,
        input_size,
        conv3x3_only=False,
        hidden_ratio=1.0,
    ):
        super(FastFlow, self).__init__()
        assert (
            backbone_name in SUPPORTED_BACKBONES
        ), "backbone_name must be one of {}".format(SUPPORTED_BACKBONES)

        if backbone_name in [BACKBONE_CAIT, BACKBONE_DEIT]:
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
            channels = [768]
            scales = [16]
        else:
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # for transformers, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm
            self.norms = nn.ModuleList()
            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(input_size / scale), int(input_size / scale)],
                        elementwise_affine=True,
                    )
                )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(input_size / scale), int(input_size / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        self.input_size = input_size

    def forward(self, x):
        self.feature_extractor.eval()
        if isinstance(
            self.feature_extractor, timm.models.vision_transformer.VisionTransformer
        ):
            x = self.feature_extractor.patch_embed(x)
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
            if self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )
            x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
            for i in range(8):  # paper Table 6. Block Index = 7
                x = self.feature_extractor.blocks[i](x)
            x = self.feature_extractor.norm(x)
            x = x[:, 2:, :]
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        elif isinstance(self.feature_extractor, timm.models.cait.Cait):
            x = self.feature_extractor.patch_embed(x)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index = 40
                x = self.feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        else:
            features = self.feature_extractor(x)
            features = [self.norms[i](feature) for i, feature in enumerate(features)]

        loss = 0
        outputs = []
        for i, feature in enumerate(features):
            output, log_jac_dets = self.nf_flows[i](feature)
            loss += torch.mean(
                0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
            )
            outputs.append(output)
        ret = {"loss": loss}

        if not self.training:
            anomaly_map_list = []
            for output in outputs:
                log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
                prob = torch.exp(log_prob)
                a_map = F.interpolate(
                    -prob,
                    size=[self.input_size, self.input_size],
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_map_list.append(a_map)
            anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
            anomaly_map = torch.mean(anomaly_map_list, dim=-1)
            ret["anomaly_map"] = anomaly_map
        return ret


########################################################################################################################
# -----------------------------------------------------------------------------------------------------------------------
# metrics
# -----------------------------------------------------------------------------------------------------------------------

from sklearn import metrics as metrics_sk

def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics_sk.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics_sk.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics_sk.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    auroc = metrics_sk.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics_sk.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }


# -----------------------------------------------------------------------------------------------------------------------
# plot segmentation images
# -----------------------------------------------------------------------------------------------------------------------

def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)

        savename0 = image_path.split("/")
        # savename = "_".join(savename0[-save_depth:-1]) + '_' + savename0[-1].split('.')[0] + '_' + str(f'{anomaly_score:.3f}') + '.' + savename0[-1].split('.')[1]
        savename = "_".join(savename0[-save_depth:-1]) + '_' + savename0[-1].split('.')[0] + '_' + str(f'{anomaly_score}') + '.' + savename0[-1].split('.')[1]
        savename = os.path.join(savefolder, savename)
        f, axes = plt.subplots(1, 2 + int(masks_provided))
        # transpose to (imagesize, imagesize, 3)
        axes[0].imshow(image.transpose(1, 2, 0))
        axes[1].imshow(mask.transpose(1, 2, 0))
        axes[2].imshow(segmentation)
        f.set_size_inches(3 * (2 + int(masks_provided)), 3)
        f.tight_layout()
        f.savefig(savename)
        plt.close()


########################################################################################################################
# -----------------------------------------------------------------------------------------------------------------------
# config
# -----------------------------------------------------------------------------------------------------------------------

# resnet18
# config = {
#     'backbone_name': 'resnet18',
#     'input_size': 256,
#     'flow_step': 8,
#     'hidden_ratio': 1.0,
#     'conv3x3_only': True
# }

# # ----------
# # wide resnet50 2
# config = {
#     'backbone_name': 'wide_resnet50_2',
#     'input_size': 256,
#     'flow_step': 8,
#     'hidden_ratio': 1.0,
#     'conv3x3_only': False
# }

# # ----------
# # cait
config = {
    'backbone_name': 'cait_m48_448',
    'input_size': 448,
    'flow_step': 20,
    'hidden_ratio': 0.16,
    'conv3x3_only': False
}


# # ----------
# # deit
# config = {
#     'backbone_name': 'deit_base_distilled_patch16_384',
#     'input_size': 384,
#     'flow_step': 20,
#     'hidden_ratio': 0.16,
#     'conv3x3_only': False
# }


########################################################################################################################
# -----------------------------------------------------------------------------------------------------------------------
# base setting
# -----------------------------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/mvtech_ad/fastflow'

data_path = '/media/kswada/MyFiles/dataset/mvtec_ad'


# ----------
# checkpoint
CHECKPOINT_DIR = os.path.join(base_path, '_fastflow_experiment_checkpoints')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

checkpoint_dir = os.path.join(CHECKPOINT_DIR, 'exp_cait')
os.makedirs(checkpoint_dir, exist_ok=True)


# -----------------------------------------------------------------------------------------------------------------------
# build model and optimizer
# -----------------------------------------------------------------------------------------------------------------------

# config = yaml.safe_load(open(args.config, "r"))

model = FastFlow(
    backbone_name=config["backbone_name"],
    flow_steps=config["flow_step"],
    input_size=config["input_size"],
    conv3x3_only=config["conv3x3_only"],
    hidden_ratio=config["hidden_ratio"],
)

print(
    "Model A.D. Param#: {}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
)


LR = 1e-3
WEIGHT_DECAY = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# data loaders
# ----------------------------------------------------------------------------------------------------------------------

name = 'mvtec'

idx = 9
mvtec_classname = _CLASSNAMES[idx]
print(f'classname: {mvtec_classname}')
dataset_name = f'{name}_{mvtec_classname}'

num_workers = 12
batch_size = 32
train_val_split = 1.0
seed = 0
resize = config['input_size']
cropsize = config['input_size']

train_dataset = MVTecDataset(
    data_path,
    classname=mvtec_classname,
    resize=resize,
    train_val_split=train_val_split,
    imagesize=cropsize,
    split=DatasetSplit.TRAIN,
    seed=seed,
    augment=True,
)

test_dataset = MVTecDataset(
    data_path,
    classname=mvtec_classname,
    resize=resize,
    imagesize=cropsize,
    split=DatasetSplit.TEST,
    seed=seed,
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
)


train_dataloader.name = name


# ----------
torch.cuda.empty_cache()

imagesize = train_dataloader.dataset.imagesize

print(f'image size: {imagesize}')


# -----------------------------------------------------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------------------------------------------------

model.cuda()


# ----------
NUM_EPOCHS = 100
# NUM_EPOCHS = 10

LOG_INTERVAL = 10
EVAL_INTERVAL = 10
CHECKPOINT_INTERVAL = 10

for epoch in range(NUM_EPOCHS):
    # ----------
    # train one epoch
    model.train()
    loss_meter = AverageMeter()
    for step, dat in enumerate(train_dataloader):
        # forward
        image_val = dat['image']
        image_val = image_val.cuda()
        ret = model(image_val)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % LOG_INTERVAL == 0 or (step + 1) == len(train_dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )

    # ----------
    if (epoch + 1) % EVAL_INTERVAL == 0:
        model.eval()
        auroc_metric = metrics.ROC_AUC()
        for dat in test_dataloader:
            data = dat['image']
            targets = dat['mask']
            data, targets = data.cuda(), targets.cuda()
            with torch.no_grad():
                ret = model(data)
            outputs = ret["anomaly_map"].cpu().detach()
            # # ----------
            # # torch.Size([batchsize, 1, input_size, input_size])
            # print(f'outputs shape: {outputs.shape}')
            # # torch.Size([batchsize, 1, input_size, input_size])
            # print(f'targets shape: {targets.shape}')
            # ----------
            outputs = outputs.flatten()
            targets = targets.flatten()
            auroc_metric.update((outputs, targets))
        auroc = auroc_metric.compute()
        print("AUROC: {}".format(auroc))

    # ----------
    if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(checkpoint_dir, "%d.pt" % epoch),
        )


# -----------------------------------------------------------------------------------------------------------------------
# evaluate
# -----------------------------------------------------------------------------------------------------------------------

# ----------
# build model
model = FastFlow(
    backbone_name=config["backbone_name"],
    flow_steps=config["flow_step"],
    input_size=config["input_size"],
    conv3x3_only=config["conv3x3_only"],
    hidden_ratio=config["hidden_ratio"],
)

print(
    "Model A.D. Param#: {}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
)


# ----------
# load checkpoint
checkpoint_path = os.path.join(checkpoint_dir, '39.pt')
checkpoint = torch.load(checkpoint_path)

model.load_state_dict(checkpoint["model_state_dict"])


# ----------
test_dataset = MVTecDataset(
    data_path,
    classname=mvtec_classname,
    resize=resize,
    imagesize=cropsize,
    split=DatasetSplit.TEST,
    seed=seed,
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
)


# ----------
# evaluate
model.cuda()

model.eval()

auroc_metric = metrics.ROC_AUC()

segmentations = []
segmentations_target = []

for dat in test_dataloader:
    data = dat['image']
    targets = dat['mask']
    data, targets = data.cuda(), targets.cuda()
    with torch.no_grad():
        ret = model(data)
    outputs = ret["anomaly_map"].cpu().detach()
    segmentations.append(outputs)
    segmentations_target.append(targets.cpu())
    # ----------
    # torch.Size([batchsize, 1, input_size, input_size])
    print(f'outputs shape: {outputs.shape}')
    # torch.Size([batchsize, 1, input_size, input_size])
    print(f'targets shape: {targets.shape}')
    # ----------
    outputs = outputs.flatten()
    targets = targets.flatten()
    auroc_metric.update((outputs, targets))


# ----------
auroc = auroc_metric.compute()

print("AUROC: {}".format(auroc))


# ----------
print(len(segmentations))
print(segmentations[0].shape)

print(len(segmentations_target))
print(segmentations_target[0].shape)


segmentations_concat = np.concatenate(segmentations, 0)
segmentations_target_concat = np.concatenate(segmentations_target, 0)
print(segmentations_concat.shape)
print(segmentations_concat[0])

# ----------
min_scores = (
    segmentations_concat.reshape(len(segmentations_concat), -1)
    .min(axis=-1)
    .reshape(-1, 1, 1, 1)
)

max_scores = (
    segmentations_concat.reshape(len(segmentations_concat), -1)
    .max(axis=-1)
    .reshape(-1, 1, 1, 1)
)

print(f'min score: {min_scores}    max score:  {max_scores}')


# ----------
segmentations_scaled = (segmentations_concat - min_scores) / (max_scores - min_scores)
# (# of test images, 1, input_size, input_size)
print(segmentations_scaled.shape)

# remove axis=1
segmentations_scaled = np.mean(segmentations_scaled, axis=1)
# now the dimension is (# of test images, input_size, input_size)
print(segmentations_scaled.shape)

segmentations_target_concat = np.mean(segmentations_target_concat, axis=1)
print(segmentations_target_concat.shape)


# ----------------------------------------------------------------------------------------------------------------------
# plot and save
#  - input test image, mask (ground truth), segmentation by each image
# ----------------------------------------------------------------------------------------------------------------------

# x[2]:  image path
image_paths = [
    x[2] for x in test_dataloader.dataset.data_to_iterate
]

# x[3]:  ground truth mask image path
mask_paths = [
    x[3] for x in test_dataloader.dataset.data_to_iterate
]

image_save_path = os.path.join(checkpoint_dir, "segmentation_images", dataset_name)
os.makedirs(image_save_path, exist_ok=True)


def image_transform(image):
    # reshape to apply each value to each channel
    in_std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    in_mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    image = test_dataloader.dataset.transform_img(image)
    return np.clip((image.numpy() * in_std + in_mean) * 255, 0, 255).astype(np.uint8)


def mask_transform(mask):
    return test_dataloader.dataset.transform_mask(mask).numpy()


plot_segmentation_images(
    savefolder=image_save_path,
    image_paths=image_paths,
    segmentations=segmentations_scaled,
    mask_paths=mask_paths,
    image_transform=image_transform,
    mask_transform=mask_transform,
    save_depth=4
)


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# compute evaluation metrics
# ----------------------------------------------------------------------------------------------------------------------

# Compute PRO score & PW Auroc for all images
# pixel_scores = compute_pixelwise_retrieval_metrics(segmentations, masks_gt)
pixel_scores = compute_pixelwise_retrieval_metrics(segmentations_scaled, segmentations_target_concat)

full_pixel_auroc = pixel_scores["auroc"]


# ----------
# Compute PRO score & PW Auroc only images with anomalies
sel_idxs = []

for i in range(len(segmentations_target_concat)):
    if np.sum(segmentations_target_concat[i]) > 0:
        sel_idxs.append(i)

pixel_scores = compute_pixelwise_retrieval_metrics(
    [segmentations_scaled[i] for i in sel_idxs],
    [segmentations_target_concat[i] for i in sel_idxs],
)

anomaly_pixel_auroc = pixel_scores["auroc"]


# ----------
print(f'full pixel auroc: {full_pixel_auroc}')
print(f'anomaly pixel auroc: {anomaly_pixel_auroc}')

