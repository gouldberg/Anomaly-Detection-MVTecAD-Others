# export LD_LIBRARY_PATH=/home/kswada/kw/mvtech_ad/patchcore/venv/lib/python3.8/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

import os
import glob
# import shutil
import argparse

import numpy as np
from PIL import Image
import cv2

from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.ndimage import gaussian_filter
from src.sampling_methods.kcenter_greedy import kCenterGreedy

import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
# from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.models as torch_models

from torchinfo import summary
import timm
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

# ----------
# PyTorch Lightining
# https://pytorch-lightning.readthedocs.io/en/latest/
import pytorch_lightning as pl

# ----------
# faiss
# https://github.com/facebookresearch/faiss
import faiss

# ----------
# All codes are based on: Unofficial implementation of PatchCore
# Original Paper : Towards Total Recall in Industrial Anomaly Detection (Jun 2021)
# https://github.com/hcw-00/PatchCore_anomaly_detection


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# distance matrix
# ----------------------------------------------------------------------------------------------------------------

def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors
    y = x if type(y) == type(None) else y
    # -----------
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    # -----------
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    # -----------
    dist = torch.pow(x - y, p).sum(2)
    return dist


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# NN, KNN
# ----------------------------------------------------------------------------------------------------------------

class NN():
    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)
    # -----------
    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y
    # -----------
    def __call__(self, x):
        return self.predict(x)
    # -----------
    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        # -----------
        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):
    def __init__(self, X=None, Y=None, k=3, p=2):
        self.k = k
        super().__init__(X, Y, p)
    # -----------
    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()
    # -----------
    def predict(self, x):
        dist = torch.cdist(x, self.train_pts, self.p)
        knn = dist.topk(self.k, largest=False)
        return knn


# ----------------------------------------------------------------------------------------------------------------
# MVTecDataset
#   - PyTorch Dataset and Dataloaders
#     https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# ----------------------------------------------------------------------------------------------------------------

class MVTecDataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1
    # -----------
    def load_dataset(self):
        # -----------
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []
        # -----------
        defect_types = os.listdir(self.img_path)
        # -----------
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))
        # -----------
        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types
    # -----------
    def __len__(self):
        return len(self.img_paths)
    # -----------
    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        # -----------
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"
        return img, gt, label, os.path.basename(img_path[:-4]), img_type


# ----------------------------------------------------------------------------------------------------------------
# PatchCore
#   - PyTorch Lightning PyTorch to PyTorch Lightning:
#     https://pytorch-lightning.readthedocs.io/en/0.7.1/introduction_guide.html
# ----------------------------------------------------------------------------------------------------------------

# imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

# note: only accept 2 layers (2 embeddings: x and y)
# x = embeddings[0]  (32, 64, 16, 16)
# y = embeddings[1]  (32, 64, 16, 16)
# final z = (32, 160, 16, 16)
def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z

def patchfy(x, y, patchsize=5):
    padding = int((patchsize - 1) / 2)
    stride = 5
    unfolder = torch.nn.Unfold(kernel_size=patchsize, stride=stride, padding=padding, dilation=1)
    unfolded_features_x = unfolder(x)
    unfolded_features_y = unfolder(y)
    unfolded_features_x = unfolded_features_x.reshape(*x.shape[:2], patchsize, patchsize, -1)
    unfolded_features_y = unfolded_features_y.reshape(*y.shape[:2], patchsize, patchsize, -1)
    unfolded_features_x = unfolded_features_x.permute(0, 3, 1, 2)
    unfolded_features_y = unfolded_features_y.permute(0, 3, 1, 2)
    # unfolded_features_x = unfolded_features_x.permute(0, 4, 1, 2, 3)
    # unfolded_features_y = unfolded_features_y.permute(0, 4, 1, 2, 3)
    unfolded_features = torch.cat((unfolded_features_x, unfolded_features_y), 1)
    return unfolded_features

unfolded_features_x2 = unfolded_features_x.reshape(*x.shape[:2], patchsize, patchsize, -1)
unfolded_features_x3 = unfolded_features_x.reshape(*x.shape[:1], patchsize, patchsize, -1)

def stable_softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    softmax = np.max(numerator) / denominator
    return softmax[0]

def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    if a_max - a_min == 0:
        return 0
    else:
        return (image-a_min)/(a_max - a_min)
    
# def fnr_fpr_score(y_true, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).flatten()
#     return fn / (fn + tp), fp / (tn + fp)


# ----------
class PatchCore(pl.LightningModule):
    def __init__(self, hparams):
        super(PatchCore, self).__init__()
        # -----------
        self.save_hyperparameters(hparams)
        self.init_features()
        def hook_t(module, input, output):
            self.features.append(output)
        # def hook_t_timm(module, input, output):
        #     self.append(output)
        # ----------
        # Wide ResNet 50
        if args.model_name == 'wideresnet50':
            self.model = torch.hub.load('pytorch/vision:v0.14.0', 'wide_resnet50_2', pretrained=True)
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.layer2[-1].register_forward_hook(hook_t)
            self.model.layer3[-1].register_forward_hook(hook_t)
        # ----------
        # MobileNetV2
        elif args.model_name == 'mobilenetv2':
            self.model = torch.hub.load('pytorch/vision:v0.14.0', 'mobilenet_v2', pretrained=True)
            for param in self.model.parameters():
                param.requires_grad = False
            # for idx in [8, 11, 12, 13, 14, 15, 16]:
            for idx in [8, 11]:
            # for idx in [7, 17]:
            # for idx in [5, 8, 11, 14]:
            # for idx in [11, 14]:
                    self.model.features[idx].register_forward_hook(hook_t)
        # ----------
        # MobileNetV3 large
        elif args.model_name == 'mobilenetv3_large':
            self.model = torch_models.mobilenet_v3_large()
            for param in self.model.parameters():
                param.requires_grad = False
            # for idx in [6, 9]:
            for idx in [8, 14]:  # this is best 0.75 for screw
            # for idx in [12, 14]:
            # for idx in [14, 16]:
                self.model.features[idx].register_forward_hook(hook_t)
        # ----------
        # MobileNetV2_100
        # elif args.model_name == 'mobilenetv2_100':
        #     model = timm.create_model('mobilenetv2_100', pretrained=True)
        #     self.model = create_feature_extractor(model, ['blocks.2.2.bn2.act', 'blocks.3.3.bn2.act'])
        #     for param in self.model.parameters():
        #         param.requires_grad = False
        #     self.model.children.register_forward_hook(hook_t_timm)
        # ----------
        # EfficientNet
        elif args.model_name == 'efficientnet':
            # self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
            # self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b0', pretrained=True)
            # self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
            self.model = torch_models.efficientnet_b1()
            # self.model = torch_models.efficientnet_b3()
            for param in self.model.parameters():
                param.requires_grad = False
            for idx in [4, 6, 8]:
                self.model.features[idx].register_forward_hook(hook_t)
        # ----------
        self.criterion = torch.nn.MSELoss(reduction='sum')
        # -----------
        self.init_results_list()
        # -----------
        self.data_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size),
                        # transforms.RandomRotation(degrees=45),
                        # transforms.RandomVerticalFlip(p=0.3),
                        # transforms.RandomHorizontalFlip(p=0.3),
                        # transforms.RandomPerspective(distortion_scale=0.2, p=0.9),
                        transforms.Normalize(mean=mean_train,
                                            std=std_train)])
        self.gt_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size)])
        # -----------
        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])
    # -----------
    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []        
    # -----------
    def init_features(self):
        self.features = []
    # -----------
    def forward(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features
    # -----------
    def save_anomaly_map(self, anomaly_map, input_img, gt_img, file_name, x_type, score):
        if anomaly_map.shape != input_img.shape:
            anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
        anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm * 255)
        # -----------
        # anomaly map on image
        heatmap = cvt2heatmap(anomaly_map_norm * 255)
        hm_on_img = heatmap_on_image(heatmap, input_img)
        # -----------
        # save images
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img_{score:.2f}.jpg'), hm_on_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)
    # -----------
    def train_dataloader(self):
        image_datasets = MVTecDataset(root=os.path.join(args.dataset_path, args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='train')
        train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        return train_loader
    # -----------
    def test_dataloader(self):
        test_datasets = MVTecDataset(root=os.path.join(args.dataset_path, args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=args.num_workers)
        return test_loader
    # -----------
    def configure_optimizers(self):
        return None
    # -----------
    def on_train_start(self):
        self.model.eval() # to stop running_var move (maybe not critical)        
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.embedding_list = []
    # -----------
    def on_test_start(self):
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.index = faiss.read_index(os.path.join(self.embedding_dir_path, 'index.faiss'))
        # if torch.cuda.is_available():
        #     # resource setting in the case of GPU
        #     res = faiss.StandardGpuResources()
        #     self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.init_results_list()
    # -----------
    def training_step(self, batch, batch_idx): # save locally aware patch features
        x, _, _, _, _ = batch
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
            # embeddings.append(feature)
        # embedding = embedding_concat(embeddings[0], embeddings[1])
        embedding = patchfy(embeddings[0], embeddings[1])
        # embedding = patchfy(embeddings[0])
        self.embedding_list.extend(reshape_embedding(np.array(embedding)))
    # -----------
    def training_epoch_end(self, outputs): 
        total_embeddings = np.array(self.embedding_list)
        # ----------
        # Random projection
        # 'auto' => Johnson-Lindenstrauss lemma
        if args.projection_type == 'auto':
            self.randomprojector = SparseRandomProjection(n_components='auto', eps=args.projection_eps)
        elif args.projection_type == 'fix':
            # self.randomprojector = SparseRandomProjection(n_components=25)
            self.randomprojector = SparseRandomProjection(n_components=128)
        self.randomprojector.fit(total_embeddings)
        # ----------
        # Coreset Subsampling
        selector = kCenterGreedy(total_embeddings, 0, 0)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0] * args.coreset_sampling_ratio))
        self.embedding_coreset = total_embeddings[selected_idx]
        # ----------
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        # ----------
        # faiss
        self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
        self.index.add(self.embedding_coreset) 
        faiss.write_index(self.index,  os.path.join(self.embedding_dir_path, 'index.faiss'))
    # -----------
    def test_step(self, batch, batch_idx): # Nearest Neighbour Search
        x, gt, label, file_name, x_type = batch
        # ----------
        # extract embedding
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
            # embeddings.append(feature)
        # embedding_ = embedding_concat(embeddings[0], embeddings[1])
        embedding_ = patchfy(embeddings[0], embeddings[1])
        # embedding_ = patchfy(embeddings[0])
        embedding_test = np.array(reshape_embedding(np.array(embedding_)))
        # ----------
        # distance
        score_patches, _ = self.index.search(embedding_test, k=args.n_neighbors)
        # ----------
        # anomaly map
        anomaly_map = score_patches[:, 0].reshape((args.anomap_size, args.anomap_size))
        anomaly_map_resized = cv2.resize(anomaly_map, (args.input_size, args.input_size))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
        # ----------
        # score
        N_b = score_patches[np.argmax(score_patches[:, 0])]
        w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
        # w = (1 - stable_softmax(N_b))
        # Image-level score
        score = w * max(score_patches[:, 0])
        # ----------
        # ground truth
        gt_np = gt.cpu().numpy()[0, 0].astype(int)
        # ----------
        self.gt_list_px_lvl.extend(gt_np.ravel())
        self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(score)
        self.img_path_list.extend(file_name)
        # ----------
        # save images:
        x = self.inv_normalize(x)
        input_x = cv2.cvtColor(x.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
        self.save_anomaly_map(anomaly_map_resized_blur, input_x, gt_np * 255, file_name[0], x_type[0], score)
    # -----------
    def test_epoch_end(self, outputs):
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        print(f"Total pixel-level auc-roc score : {pixel_auc}")
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        print(f"Total image-level auc-roc score : {img_auc}")
        # img_fnr, img_fpr = fnr_fpr_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        # print(f"Total image-level false negative rate : {img_fnr}")
        # print(f"Total image-level false positive rate : {img_fpr}")
        print('test_epoch_end')
        # values = {'pixel_auc': pixel_auc, 'img_auc': img_auc, 'img_fnr': img_fnr, 'img_fpr': img_fpr}
        values = {'pixel_auc': pixel_auc, 'img_auc': img_auc}
        self.log_dict(values)


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# base config
# ----------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='ANOMALYDETECTION')

parser.add_argument('--phase', choices=['train', 'test'], default='train')
parser.add_argument('--dataset_path', default=r'./MVTec')
parser.add_argument('--category', default='bottle')
parser.add_argument('--num_epochs', default=1)
parser.add_argument('--batch_size', default=32)

# resize
parser.add_argument('--load_size', default=256)
# crop size
parser.add_argument('--input_size', default=224)

parser.add_argument('--model_name', type=str, default='wideresnet50')
parser.add_argument('--anomap_size', type=int, default=28)

parser.add_argument('--coreset_sampling_ratio', default=0.001)
parser.add_argument('--project_root_path', default=r'./test')
parser.add_argument('--save_src_code', default=True)
parser.add_argument('--save_anomaly_map', default=True)

parser.add_argument('--n_neighbors', type=int, default=9)

parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=12)


# ----------
# Parameter to control the quality of the embedding according to the Johnson-Lindenstrauss lemma
# when n_components is set to ‘auto’. This value should be strictly positive.
# Smaller values lead to better embedding and higher number of dimensions (n_components) in the target projection space.
parser.add_argument('--projection_type', type=str, default='auto')
parser.add_argument('--projection_eps', type=float, default=0.9)

args = parser.parse_args()


# ----------------------------------------------------------------------------------------------------------------
# original config
# ----------------------------------------------------------------------------------------------------------------

args.dataset_path = r'/media/kswada/MyFiles/dataset/mvtec_ad'


# ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

# print(device)
args.gpus = 1
args.gpus = 0
args.num_workers = 12


# ---------------------
def prep_dirs(root):
    # ----------
    # 1. make embeddings dir
    # embeddings_path = os.path.join('./04_output/', 'embeddings_wideresnet50', args.category)
    embeddings_path = os.path.join('./04_output/', 'embeddings_v2_tmp', args.category)
    # embeddings_path = os.path.join('./04_output/', 'embeddings_mobilenetv3_large', args.category)
    # embeddings_path = os.path.join('./04_output/', 'embeddings_efficientnet', args.category)
    # embeddings_path = os.path.join('./04_output/', 'embeddings_efficientnetb3', args.category)
    os.makedirs(embeddings_path, exist_ok=True)
    # ----------
    # 2. make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    # ----------
    # 3. make source code record dir & copy
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    return embeddings_path, sample_path, source_code_save_path


# ---------------------
# Wide ResNet 50-2
# ---------------------
args.model_name = 'wideresnet50'
# args.anomap_size = 28
args.anomap_size = 25
args.project_root_path = r'./04_output/test_wideresnet50_tmp'
args.projection_type = 'auto'
args.projection_eps = 0.9

# tunable
# args.num_epochs = 2
# args.n_neighbors = 3
# args.coreset_sampling_ratio = 0.001

args.num_epochs = 1
args.n_neighbors = 3
args.coreset_sampling_ratio = 0.01


# ---------------------
# MobileNet v2
# ---------------------
# args.model_name = 'mobilenetv2'
# args.project_root_path = r'./04_output/test_mobilenetv2_tmp'
#
# # patchsize = anomap_size
# args.anomap_size = 16
# args.projection_type = 'fix'
# # args.projection_type = 'auto'
# # args.projection_eps = 0.9
#
# # tunable
# args.num_epochs = 2
# args.n_neighbors = 3
# args.coreset_sampling_ratio = 0.01
# # args.coreset_sampling_ratio = 0.001
# # args.coreset_sampling_ratio = 0.1
#
# args.load_size = 256
# args.input_size = 256


# ---------------------
# MobileNet v2 100 (timm)
# ---------------------
# args.model_name = 'mobilenetv2_100'
# args.project_root_path = r'./04_output/test_mobilentv2_100_3'
# args.anomap_size = 14
# args.projection_type = 'fix'
#
# # tunable
# args.num_epochs = 3
# args.n_neighbors = 3
# args.coreset_sampling_ratio = 0.01


# ---------------------
# MobileNet v3 Large
# ---------------------
# args.model_name = 'mobilenetv3_large'
# args.project_root_path = r'./04_output/test_mobilentv3_large'
# args.anomap_size = 14
# args.projection_type = 'fix'
#
# # tunable
# args.num_epochs = 3
# args.n_neighbors = 3
# args.coreset_sampling_ratio = 0.001


# ---------------------
# EfficientNet b1
# ---------------------
# args.model_name = 'efficientnet'
# args.project_root_path = r'./04_output/test_efficientnet'
# args.anomap_size = 14
# args.projection_type = 'fix'
#
# # tunable
# args.num_epochs = 3
# args.n_neighbors = 3
# args.coreset_sampling_ratio = 0.01


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# train and test
# ----------------------------------------------------------------------------------------------------------------

# this is all categories
# category_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
#                  'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

# selected categories
category_list = ['screw']

for cat in category_list:
    args.category = cat
    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=os.path.join(args.project_root_path, args.category),
        max_epochs=args.num_epochs,
        gpus=args.gpus)
    model = PatchCore(hparams=args)
    # ----------
    trainer.fit(model)
    # ----------
    trainer.test(model)


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# check data loader and check step by step
# ----------------------------------------------------------------------------------------------------------------

data_transforms = transforms.Compose([
    transforms.Resize((args.load_size, args.load_size), Image.ANTIALIAS),
    transforms.ToTensor(),
    transforms.CenterCrop(args.input_size),
    transforms.Normalize(mean=mean_train,
                         std=std_train)])

gt_transforms = transforms.Compose([
    transforms.Resize((args.load_size, args.load_size)),
    transforms.ToTensor(),
    transforms.CenterCrop(args.input_size)])

image_datasets = MVTecDataset(root=os.path.join(args.dataset_path, args.category),
                              transform=data_transforms,
                              gt_transform=gt_transforms,
                              phase='train')

train_loader = DataLoader(image_datasets,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.num_workers)

# ----------
# get 1 batch
dat = next(iter(train_loader))

# 5 data
print(len(dat))

# transformed input image: (32, 3, 224, 224) = (batch_size, 3, input_size, input_size)
print(dat[0].shape)

# ground truth (mask): (32, 1, 224, 224) = (batch_size, 1, input_size, input_size)
print(dat[1].shape)

# label: only good (=0)
print(dat[2])
print(dat[4])

# sampled image number
print(dat[3])


# ----------
# Wide ResNet 50
model = torch.hub.load('pytorch/vision:v0.14.0', 'wide_resnet50_2', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.14.0', 'mobilenet_v2', pretrained=True)

# total parameters:  68,883,240
# layer2 (Sequential: 1-6):  output is (32, 512, 28, 28)
# layer3 (Sequential: 1-7):  output is (32, 1024, 14, 14)
print(summary(
    model,
    input_size=(args.batch_size, 3, args.input_size, args.input_size),
    col_names=["output_size", "num_params"],
))

# name and modules
for (name, module) in model.named_modules():
     print(name, module)


# you can see real value of each parameters
# for param in model.parameters():
#     print(param)

layers = list(model.children())
print(len(layers))

# layers for feature extraction
print(model.layer2[-1])
print(model.layer3[-1])


# ----------
# Here PatchCore instance
patchcore_model = PatchCore(hparams=args)


# ----------
# feature extraction for 1 batch
features = patchcore_model(dat[0])

# 2
print(len(features))

# layer2: (32, 512, 28, 28)
# mobilenetv2: (32, 64, 16, 16)
print(features[0].shape)
# layer2: (32, 1024, 14, 14)
# mobilenetv2: (32, 96, 14, 14)
print(features[1].shape)


# ----------
# average pooling --> concatenate

# kernel_size = 3, stride = 1, padding = 1
m = torch.nn.AvgPool2d(3, 1, 1)
# a = torch.randn(1,3,3)
# m(a)

embeddings = []
for feature in features:
    embeddings.append(m(feature))

print(embeddings[0].shape)
print(embeddings[1].shape)

# embedding = embedding_concat(embeddings[0], embeddings[1])
embedding = patchfy(embeddings[0], embeddings[1])

# (32, 1536, 28, 28)  (1536 = 512 + 1024)
# (32, 160, 16, 16)  (160 = 64 + 96)
print(embedding.shape)

# ----------
print(len(reshape_embedding(np.array(embedding))))
embedding_list = []
embedding_list.extend(reshape_embedding(np.array(embedding)))

print(embedding.shape[0])
print(embedding.shape[2])
print(embedding.shape[3])

print(embedding[0, :, 0, 0].shape)
print(embedding[0, :, 0, 0])

embedding_list = []
for k in range(embedding.shape[0]):
    for i in range(embedding.shape[2]):
        for j in range(embedding.shape[3]):
            embedding_list.append(embedding[k, :, i, j])

# 25088 = 32 * 28 * 28 (= batch_size * patch size * path size)
# 8192 = 32 * 16 * 16
print(len(embedding_list))
# 1536
# 160
print(len(embedding_list[0]))


###################
# EPOCH END
total_embeddings = np.array(embedding_list)

# ----------
# Random projection
randomprojector = SparseRandomProjection(n_components='auto', eps=0.9)
# randomprojector = SparseRandomProjection(n_components=128)
randomprojector.fit(total_embeddings)

# ----------
# Coreset Subsampling
selector = kCenterGreedy(total_embeddings, 0, 0)
selected_idx = selector.select_batch(model=randomprojector, already_selected=[],
                                     N=int(total_embeddings.shape[0] * args.coreset_sampling_ratio))
embedding_coreset = total_embeddings[selected_idx]

# (25088, 1536) --> (25, 1536)
# (8192, 160) --> (8, 160)
print('initial embedding size : ', total_embeddings.shape)
print('final embedding size : ', embedding_coreset.shape)

# faiss
index = faiss.IndexFlatL2(embedding_coreset.shape[1])

# ----------
# distance
print(len(embedding_list))
print(embedding_list[0].shape)
score_patches, _ = index.search(np.array(reshape_embedding(np.array(embedding))), k=args.n_neighbors)
# score_patches, _ = index.search(np.array(reshape_embedding(np.array(embedding))), k=10)

# (25088, 3)
# (8192, 3)
print(score_patches.shape)
print(score_patches)

# ----------
# anomaly map
# only 1 image
# anomaly_map = score_patches[:28*28, 0].reshape((28, 28))
# anomaly_map = score_patches[:7*7, 0].reshape((7, 7))
anomaly_map = score_patches[:5*5, 0].reshape((5, 5))
anomaly_map_resized = cv2.resize(anomaly_map, (args.input_size, args.input_size))
anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)

# ----------
# score
# N_b = score_patches[np.argmax(score_patches[:28*28, 0])]
N_b = score_patches[np.argmax(score_patches[:7*7, 0])]

# w = (1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b))))
w = (1 - stable_softmax(N_b))

# score = w * max(score_patches[:28*28, 0])  # Image-level score
# score = w * max(score_patches[:16*16, 0])  # Image-level score
score = w * max(score_patches[:7*7, 0])  # Image-level score


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# check mobilenet v2
# ----------------------------------------------------------------------------------------------------------------

model = torch.hub.load('pytorch/vision:v0.14.0', 'mobilenet_v2', pretrained=True)

for (name, module) in model.named_modules():
    #  print(name, module)
     print(name)

# 19
print(len(model.features))

for i in range(len(model.features)):
    print(model.features[i])

# 3.5M parameters
print(summary(
    model,
    input_size=(args.batch_size, 3, args.input_size, args.input_size),
    col_names=["output_size", "num_params"],
))


# ----------------------------------------------------------------------------------------------------------------
# check mobilenet v2 by timm
# ----------------------------------------------------------------------------------------------------------------

model = timm.create_model('mobilenetv2_100', pretrained=True)

for (name, module) in model.named_modules():
    #  print(name, module)
     print(name)

# 3.5M parameters
print(summary(
    model,
    input_size=(args.batch_size, 3, args.input_size, args.input_size),
    col_names=["output_size", "num_params"],
))


# ----------
train_nodes, eval_nodes = get_graph_node_names(model)

extractor = create_feature_extractor(model, ['blocks.2.2.bn2.act', 'blocks.3.3.bn2.act'])

print(summary(
    extractor,
    input_size=(args.batch_size, 3, args.input_size, args.input_size),
    col_names=["output_size", "num_params"],
))


# ----------------------------------------------------------------------------------------------------------------
# check mobilenet v3
# ----------------------------------------------------------------------------------------------------------------

# model = torch_models.mobilenet_v3_small()
model = torch_models.mobilenet_v3_large()

for (name, module) in model.named_modules():
    #  print(name, module)
     print(name)

# small: 13
# large: 17
print(len(model.features))

for i in range(len(model.features)):
    print(model.features[i])

# small: 2.5M parameters
# large: 5.5M parameters
print(summary(
    model,
    input_size=(args.batch_size, 3, args.input_size, args.input_size),
    col_names=["output_size", "num_params"],
))



##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# check efficient net
# ----------------------------------------------------------------------------------------------------------------

# model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
# model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b0', pretrained=True)
# model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
# model = torch_models.efficientnet_b1()
model = torch_models.efficientnet_b3()

for (name, module) in model.named_modules():
    #  print(name, module)
     print(name)

print(len(model.features))

for i in range(len(model.features)):
    print(model.features[i])

# total parameters (b0): 5,288,548
# total parameters (widese b0): 8,423,848
# total parameters (b4): 19.341.616
# total parameters (b1): 7,794,184
print(summary(
    model,
    input_size=(args.batch_size, 3, args.input_size, args.input_size),
    col_names=["output_size", "num_params"],
))

