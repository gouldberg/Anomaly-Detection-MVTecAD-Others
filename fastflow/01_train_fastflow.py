
import os
from glob import glob

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm

from PIL import Image

import yaml
from ignite.contrib import metrics

import FrEIA.framework as Ff
import FrEIA.modules as Fm


########################################################################################################################
# -----------------------------------------------------------------------------------------------------------------------
# constants
# -----------------------------------------------------------------------------------------------------------------------

MVTEC_CATEGORIES = [
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

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, mvtec_cat, input_size, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, mvtec_cat, "train", "good", "*.png")
            )
        else:
            self.image_files = glob(os.path.join(root, mvtec_cat, "test", "*", "*.png"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        # ----------
        image = Image.open(image_file).convert("RGB")
        # image = Image.open(image_file)
        # ----------
        image_trans = self.image_transform(image)
        if self.is_train:
            return image_trans
        else:
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image_trans.shape[-2], image_trans.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("/test/", "/ground_truth/").replace(
                        ".png", "_mask.png"
                    )
                )
                target = self.target_transform(target)
            return image_trans, target

    def __len__(self):
        return len(self.image_files)


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
# config
# -----------------------------------------------------------------------------------------------------------------------

# resnet18
config = {
    'backbone_name': 'resnet18',
    'input_size': 256,
    'flow_step': 8,
    'hidden_ratio': 1.0,
    'conv3x3_only': True
}

# # ----------
# # wide resnet50 2
# config = {
#     'backbone_name': 'wide_resnet50_2',
#     'input_size': 256,
#     'flow_step': 8,
#     'hidden_ratio': 1.0,
#     'conv3x3_only': False
# }
#
# # ----------
# # cait
# config = {
#     'backbone_name': 'cait_m48_448',
#     'input_size': 448,
#     'flow_step': 20,
#     'hidden_ratio': 0.16,
#     'conv3x3_only': False
# }
#
#
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

checkpoint_dir = os.path.join(
    CHECKPOINT_DIR, "exp%d" % len(os.listdir(CHECKPOINT_DIR))
)

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


# -----------------------------------------------------------------------------------------------------------------------
# data loader
# -----------------------------------------------------------------------------------------------------------------------

print(MVTEC_CATEGORIES)
mvtec_category = 'screw'


# ----------
BATCH_SIZE = 32

train_dataset = MVTecDataset(
    root=data_path,
    mvtec_cat=mvtec_category,
    input_size=config["input_size"],
    is_train=True,
)

test_dataset = MVTecDataset(
    root=data_path,
    mvtec_cat=mvtec_category,
    input_size=config["input_size"],
    is_train=False,
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1,
    drop_last=True,
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=1,
    drop_last=False,
)


# ---------
# print(train_dataset.image_files)
# print(train_dataset.image_transform)
# print(len(train_dataset))
#
# idx = 0
# output = train_dataset[idx]
# print(output)
#
# tmp = next(iter(train_dataloader))
# print(tmp)
#

# -----------------------------------------------------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------------------------------------------------

model.cuda()


# ----------
# NUM_EPOCHS = 500
NUM_EPOCHS = 10

LOG_INTERVAL = 10
EVAL_INTERVAL = 10
CHECKPOINT_INTERVAL = 10

for epoch in range(NUM_EPOCHS):
    # ----------
    # train one epoch
    model.train()
    loss_meter = AverageMeter()
    for step, data in enumerate(train_dataloader):
        # forward
        data = data.cuda()
        ret = model(data)
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
        for data, targets in test_dataloader:
            # data: (32, 3, 256, 256)   targets: (32, 1, 256 ,256)
            data, targets = data.cuda(), targets.cuda()
            with torch.no_grad():
                ret = model(data)
            outputs = ret["anomaly_map"].cpu().detach()
            # ----------
            # torch.Size([batchsize, 1, input_size, input_size])
            print(f'outputs shape: {outputs.shape}')
            # torch.Size([batchsize, 1, input_size, input_size])
            print(f'targets shape: {targets.shape}')
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
checkpoint_path = os.path.join(checkpoint_dir, '9.pt')
checkpoint = torch.load(checkpoint_path)

model.load_state_dict(checkpoint["model_state_dict"])


# ----------
# evaluate
model.cuda()

model.eval()

auroc_metric = metrics.ROC_AUC()

segmentations = []

for data, targets in test_dataloader:
    data, targets = data.cuda(), targets.cuda()
    with torch.no_grad():
        ret = model(data)
    outputs = ret["anomaly_map"].cpu().detach()
    segmentations.append(outputs)
    # ----------
    # torch.Size([batchsize, 1, input_size, input_size])
    print(f'outputs shape: {outputs.shape}')
    # torch.Size([batchsize, 1, input_size, input_size])
    print(f'targets shape: {targets.shape}')
    # ----------
    outputs = outputs.flatten()
    targets = targets.flatten()
    auroc_metric.update((outputs, targets))

auroc = auroc_metric.compute()

print("AUROC: {}".format(auroc))


# ----------
print(len(segmentations))
print(segmentations[0].shape)

segmentations_concat = np.concatenate(segmentations, 0)
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
