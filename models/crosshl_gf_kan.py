import sys
sys.path.append("./../")
import os
import numpy as np
import random
from torch import einsum
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data as dataf
from torch.utils.data import Dataset
from scipy import io
from scipy.io import loadmat as loadmat
from sklearn.decomposition import PCA
from torch.nn.parameter import Parameter
import torchvision.transforms.functional as TF
from torch.nn import LayerNorm,Linear,Dropout,Softmax
import time
from PIL import Image
import math
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from torchsummary import summary
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import re
from pathlib import Path
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

cudnn.deterministic = True
cudnn.benchmark = False
import numpy as np
import torch
from operator import truediv


def log_result(oa_ae, aa_ae, kappa_ae, element_acc_ae, path):
    f = open(path, 'w')
    sentence0 = 'OAs for each iteration are:' + str(oa_ae) + '\n'
    f.write(sentence0)
    sentence1 = 'AAs for each iteration are:' + str(aa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'KAPPAs for each iteration are:' + str(kappa_ae) + '\n' + '\n'
    f.write(sentence2)
    sentence3 = 'mean_OA ± std_OA is: ' + str(np.mean(oa_ae)) + ' ± ' + str(np.std(oa_ae)) + '\n'
    f.write(sentence3)
    sentence4 = 'mean_AA ± std_AA is: ' + str(np.mean(aa_ae)) + ' ± ' + str(np.std(aa_ae)) + '\n'
    f.write(sentence4)
    sentence5 = 'mean_KAPPA ± std_KAPPA is: ' + str(np.mean(kappa_ae)) + ' ± ' + str(np.std(kappa_ae)) + '\n' + '\n'
    f.write(sentence5)

    element_mean = np.mean(element_acc_ae, axis=0)
    element_std = np.std(element_acc_ae, axis=0)
    sentence8 = "Mean of all elements in confusion matrix: " + str(element_mean) + '\n'
    f.write(sentence8)
    sentence9 = "Standard deviation of all elements in confusion matrix: " + str(element_std) + '\n' + '\n'
    f.write(sentence9)
    element_mean = list(element_mean)
    element_mean.extend([np.mean(oa_ae),np.mean(aa_ae),np.mean(kappa_ae)])
    element_std = list(element_std)
    element_std.extend([np.std(oa_ae),np.std(aa_ae),np.std(kappa_ae)])
    sentence10 = "All values without std: " + str(element_mean) + '\n' + '\n'
    f.write(sentence10)
    sentence11 = "All values with std: "
    for i,x in enumerate(element_mean):
        sentence11 += str(element_mean[i]) + " ± " +  str(element_std[i]) + ", "
    sentence11 += "\n"
    f.write(sentence11)
    f.close()
    
##Mere vala for trento

def result_reports(xtest, xtest2, ytest, name, model, iternum):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_pred = np.empty((len(ytest)), dtype=np.float32)
    number = len(ytest) // test_batch_size

    model.to(device)
    xtest, xtest2, ytest = xtest.to(device), xtest2.to(device), ytest.to(device)

    for i in range(number):
        temp = xtest[i * test_batch_size:(i + 1) * test_batch_size, :, :].to(device)
        temp1 = xtest2[i * test_batch_size:(i + 1) * test_batch_size, :, :].to(device)
        temp2 = model(temp, temp1)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        y_pred[i * test_batch_size:(i + 1) * test_batch_size] = temp3.cpu().numpy()
        del temp, temp1, temp2, temp3

    if (i + 1) * test_batch_size < len(ytest):
        temp = xtest[(i + 1) * test_batch_size:len(ytest), :, :].to(device)
        temp1 = xtest2[(i + 1) * test_batch_size:len(ytest), :, :].to(device)
        temp2 = model(temp, temp1)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        y_pred[(i + 1) * test_batch_size:len(ytest)] = temp3.cpu().numpy()
        del temp, temp1, temp2, temp3

    y_pred = torch.from_numpy(y_pred).long().to(device)

    overall_acc = accuracy_score(ytest.cpu(), y_pred.cpu())
    confusion_mat = confusion_matrix(ytest.cpu(), y_pred.cpu())
    class_acc, avg_acc = AvgAcc_andEachClassAcc(confusion_mat)
    kappa_score = cohen_kappa_score(ytest.cpu(), y_pred.cpu())
    createConfusionMatrix(ytest.cpu(), y_pred.cpu(), f"{name}_test_{iternum}")

    return confusion_mat, overall_acc * 100, class_acc * 100, avg_acc * 100, kappa_score * 100

def createConfusionMatrix(y_test, y_pred, plt_name):
    df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(6), range(6))
    df_cm.columns = ['Buildings', 'Woods', 'Roads', 'Apples', 'Ground', 'Vineyard']
    df_cm = df_cm.rename({0: 'Buildings', 1: 'Woods', 2: 'Roads', 3: 'Apples', 4: 'Ground', 5: 'Vineyard'})
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.set(font_scale=0.9)
    plt.figure(figsize=(30, 30))
    sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')
    plt.savefig(f'Cross-HL_{plt_name}.eps', format='eps')

def AvgAcc_andEachClassAcc(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    class_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(class_acc)
    return class_acc, average_acc

# Assuming `model`, `test_patch_hsi`, `test_patch_lidar`, and `test_label` are defined elsewhere in your script


class HSI_LiDAR_DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, dataset='Trento'):

        HSI = loadmat(f'/content/drive/MyDrive/{dataset}11x11/HSI_Tr.mat')
        LiDAR = loadmat(f'/content/drive/MyDrive/{dataset}11x11/LIDAR_Tr.mat')
        label = loadmat(f'/content/drive/MyDrive/{dataset}11x11/TrLabel.mat')

        self.hs_image = (torch.from_numpy(HSI['Data'].astype(np.float32)).to(torch.float32)).permute(0,3,1,2)
        self.lidar_image = (torch.from_numpy(LiDAR['Data'].astype(np.float32)).to(torch.float32)).permute(0,3,1,2)
        self.lbls = ((torch.from_numpy(label['Data'])-1).long()).reshape(-1)

    def __len__(self):
        return self.hs_image.shape[0]

    def __getitem__(self, i):
        return self.hs_image[i], self.lidar_image[i], self.lbls[i]

class HSI_LiDAR_DatasetTest(torch.utils.data.Dataset):
    def __init__(self, dataset='Trento'):
        HSI = loadmat(f'/content/drive/MyDrive/{dataset}11x11/HSI_Te.mat')
        LiDAR = loadmat(f'/content/drive/MyDrive/{dataset}11x11/LIDAR_Te.mat')
        label = loadmat(f'/content/drive/MyDrive/{dataset}11x11/TeLabel.mat')

        self.hs_image = (torch.from_numpy(HSI['Data'].astype(np.float32)).to(torch.float32)).permute(0,3,1,2)
        self.lidar_image = (torch.from_numpy(LiDAR['Data'].astype(np.float32)).to(torch.float32)).permute(0,3,1,2)
        self.lbls = ((torch.from_numpy(label['Data'])-1).long()).reshape(-1)


    def __len__(self):
        return self.hs_image.shape[0]

    def __getitem__(self, i):
        return self.hs_image[i], self.lidar_image[i], self.lbls[i]
    
import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features

        # Store the original shape of the input tensor
        original_shape = x.shape

        # Reshape the input tensor to (some, in_features)
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )

        # Reshape the output tensor back to (*, out_features)
        output = base_output + spline_output
        output = output.view(*original_shape[:-1], self.out_features)

        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

## For GPU
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as dataf
import time
from torchsummary import summary
from torch.nn import LayerNorm, Linear, Dropout, Softmax
from torch.nn.modules.container import Sequential
import copy
import torch.nn.functional as F
import torch.fft
import math
from functools import partial
import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=None, p=64, g=64):
        super(HetConv, self).__init__()
        self.groupwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, groups=g, padding=kernel_size//3, stride=stride)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=p, stride=stride)
    def forward(self, x):
        return self.groupwise_conv(x) + self.pointwise_conv(x)

class CrossHL_attention(nn.Module):
    def __init__(self, dim, patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.dim = dim
        self.Wq = nn.Linear(patches, dim * num_heads, bias=qkv_bias)
        self.Wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wv = nn.Linear(patches, dim, bias=qkv_bias)
        self.linear_projection = nn.Linear(dim * num_heads, dim)
        self.linear_projection_drop = nn.Dropout(proj_drop)

    def forward(self, x, x2):
        B, N, C = x.shape
        query = self.Wq(x2).reshape(B, self.num_heads, self.num_heads, self.dim // self.num_heads).permute(0, 1, 2, 3)
        key = self.Wk(x).reshape(B, N, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        value = self.Wv(x.transpose(1,2)).reshape(B, C, self.num_heads, self.dim // self.num_heads).permute(0, 2, 3, 1)
        attention = torch.einsum('bhid,bhjd->bhij', key, query) * self.scale
        attention = attention.softmax(dim=-1)
        x = torch.einsum('bhij,bhjd->bhid', attention, value)
        x = x.reshape(B, N, -1)
        x = self.linear_projection(x)
        x = self.linear_projection_drop(x)
        return x

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size
        x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = nn.Parameter(torch.randn(x.shape) * 0.02).to(device)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        return x

class MultiLayerPerceptron(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.fclayer1 = Linear(dim, mlp_dim)
        self.fclayer2 = Linear(mlp_dim, dim)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fclayer1.weight)
        nn.init.xavier_uniform_(self.fclayer2.weight)
        nn.init.normal_(self.fclayer1.bias, std=1e-6)
        nn.init.normal_(self.fclayer2.bias, std=1e-6)

    def forward(self, x):
        x = self.fclayer1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fclayer2(x)
        x = self.dropout(x)
        return x

class SingleEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim):
        super().__init__()
        self.attention_norm = LayerNorm(dim, eps=1e-6)
        self.ffn_norm = LayerNorm(dim, eps=1e-6)
        self.cross_hl_attention = CrossHL_attention(dim=dim, patches=11**2)
        self.Globalfilter = GlobalFilter(dim, h=14, w=8)
        self.KAN=KAN(layers_hidden=[64, 64, 64])
    def forward(self, x1, x2):
        res = x1
        x = self.attention_norm(x1)
        x_gf=self.Globalfilter(x)
        x_chl= self.cross_hl_attention(x,x2)
        # print(f"Cross attention layer executed with shape as {x.shape}")
        x_chl = x_chl + res
        # x = x + res
        res = x
        x_chl = self.ffn_norm(x_chl)
        x_gf= self.ffn_norm(x_gf)
        # print(x_gf.shape,x_chl.shape)
        x=x_chl+x_gf
        # print(f"Shape before res{res.shape}")
        # print(f"Shape before KAN{x.shape}")
        x = self.KAN(x)
        # print(f"Shape after KAN{x.shape}")

        x = x + res
        return x

class Encoder(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_dim=512, depth=2):
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(dim, eps=1e-6)
        for _ in range(depth):
            layer = SingleEncoderBlock(dim, num_heads, mlp_dim)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x, x2):
        for layer_block in self.layer:
            x = layer_block(x, x2)
        encoded = self.encoder_norm(x)
        return encoded[:, 0]

class CrossHL_Transformer(nn.Module):
    def __init__(self, FM, NC, NCLidar, Classes, patchsize):
        super(CrossHL_Transformer, self).__init__()
        self.patchsize = patchsize
        self.NCLidar = NCLidar
        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 8, (9, 3, 3), padding=(0, 1, 1), stride=1),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.hetconv_layer = nn.Sequential(
            HetConv(8 * (NC - 8), FM*4, p=1, g=(FM*4)//4 if (8 * (NC - 8))%FM == 0 else (FM*4)//8),
            nn.BatchNorm2d(FM*4),
            nn.ReLU()
        )
        self.ca = Encoder(FM*4)
        self.fclayer = nn.Linear(FM*4, Classes)
        self.position_embeddings = nn.Parameter(torch.randn(1, (patchsize**2), FM*4))
        self.dropout = nn.Dropout(0.1)
        torch.nn.init.xavier_uniform_(self.fclayer.weight)
        torch.nn.init.normal_(self.fclayer.bias, std=1e-6)

    def forward(self, x1, x2):
        x1 = x1.reshape(x1.shape[0], -1, self.patchsize, self.patchsize).unsqueeze(1).to(device)
        x2 = x2.reshape(x1.shape[0], -1, self.patchsize*self.patchsize).to(device)
        if x2.shape[1] > 0:
            x2 = F.adaptive_avg_pool1d(x2.flatten(2).transpose(1, 2), 1).transpose(1, 2).reshape(x1.shape[0], -1, self.patchsize*self.patchsize)
        x1 = self.conv5(x1)
        x1 = x1.reshape(x1.shape[0], -1, self.patchsize, self.patchsize)
        x1 = self.hetconv_layer(x1)
        x1 = x1.flatten(2).transpose(-1, -2)
        x = x1 + self.position_embeddings
        x = self.dropout(x)
        x = self.ca(x, x2)
        x = x.reshape(x.shape[0], -1)
        out = self.fclayer(x)
        return out

# Dataset and other parts of the code where data is loaded and moved to GPU should be updated accordingly.


## Training code for T4GPU
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as dataf
import time
from torchsummary import summary
# import utils
# import logger

datasetNames = ["Trento"]  # ["Trento", "MUUFL", "Houston"]
MultiModalData = 'LiDAR'
modelName = 'Cross-HL'

patchsize = 11
batch_size = 64  # batch size for training
test_batch_size = 500
EPOCHS = 20
learning_rate = 5e-4
FM = 16
FileName = 'CrossHL'
num_heads = 8  # d_h = number of mhsa heads
mlp_dim = 512
depth = 2  # Number of transformer encoder layer
num_iterations = 2
train_loss = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_val(seed=14):
    torch.manual_seed(seed)
    np.random.seed(seed)

for dataset in datasetNames:
    print(f"---------------------------------- Details for {dataset} dataset---------------------------------------------")
    print('\n')
    try:
        os.makedirs(dataset)
    except FileExistsError:
        pass

    # Assuming HSI_LiDAR_DatasetTrain and HSI_LiDAR_DatasetTest are defined elsewhere
    train_dataset = HSI_LiDAR_DatasetTrain(dataset=dataset)
    test_dataset = HSI_LiDAR_DatasetTest(dataset=dataset)

    NC = train_dataset.hs_image.shape[1]
    NCLidar = train_dataset.lidar_image.shape[1]
    Classes = len(torch.unique(train_dataset.lbls))

    train_loader = dataf.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_patch_hsi = test_dataset.hs_image.to(device)
    test_patch_lidar = test_dataset.lidar_image.to(device)
    test_label = test_dataset.lbls.to(device)

    KAPPA = []
    OA = []
    AA = []
    ELEMENT_ACC = np.zeros((num_iterations, Classes))  # num_iterationsxNC

    seed_val(14)
    for iterNum in range(num_iterations):
        print('\n')
        print("---------------------------------- Summary ---------------------------------------------")
        print('\n')
        model = CrossHL_Transformer(FM=FM, NC=NC, NCLidar=NCLidar, Classes=Classes, patchsize=patchsize).to(device)
        summary(model, [(NC, patchsize**2), (NCLidar, patchsize**2)])

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-3)
        loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
        BestAcc = 0

        print('\n')
        print(f"---------------------------------- Training started for {dataset} dataset ---------------------------------------------")
        print('\n')
        start = time.time()

        for epoch in range(EPOCHS):
            for step, (batch_hsi, batch_ldr, batch_lbl) in enumerate(train_loader):
                batch_hsi = batch_hsi.to(device)
                batch_ldr = batch_ldr.to(device)
                batch_lbl = batch_lbl.to(device)
                out = model(batch_hsi, batch_ldr)
                loss = loss_func(out, batch_lbl)
                optimizer.zero_grad()  # Clearing gradients
                loss.backward()
                optimizer.step()

                if step % 50 == 0:
                    model.eval()
                    y_pred = np.empty((len(test_label)), dtype='float32')
                    number = len(test_label) // test_batch_size
                    for i in range(number):
                        temp = test_patch_hsi[i * test_batch_size:(i + 1) * test_batch_size, :, :]
                        temp1 = test_patch_lidar[i * test_batch_size:(i + 1) * test_batch_size, :, :]
                        temp2 = model(temp, temp1)
                        temp3 = torch.max(temp2, 1)[1].squeeze()
                        y_pred[i * test_batch_size:(i + 1) * test_batch_size] = temp3.cpu()
                    if (i + 1) * test_batch_size < len(test_label):
                        temp = test_patch_hsi[(i + 1) * test_batch_size:len(test_label), :, :]
                        temp1 = test_patch_lidar[(i + 1) * test_batch_size:len(test_label), :, :]
                        temp2 = model(temp, temp1)
                        temp3 = torch.max(temp2, 1)[1].squeeze()
                        y_pred[(i + 1) * test_batch_size:len(test_label)] = temp3.cpu()

                    y_pred = torch.from_numpy(y_pred).long().to(device)
                    accuracy = torch.sum(y_pred == test_label).type(torch.FloatTensor) / test_label.size(0)

                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.4f' % (accuracy*100))
                    train_loss.append(loss.data.cpu().numpy())

                    if accuracy > BestAcc:
                        BestAcc = accuracy
                        torch.save(model.state_dict(), dataset+'/net_params_'+FileName+'.pkl')

                    model.train()
            scheduler.step()
        end = time.time()
        print('\nThe train time (in seconds) is:', end - start)
        Train_time = end - start

        model.load_state_dict(torch.load(dataset+'/net_params_'+FileName+'.pkl'))
        model.to(device)
        model.eval()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model.to(device)
        test_patch_hsi, test_patch_lidar, test_label = test_patch_hsi.to(device), test_patch_lidar.to(device), test_label.to(device)

        confusion_mat, overall_acc, class_acc, avg_acc, kappa_score = result_reports(test_patch_hsi, test_patch_lidar, test_label, dataset, model, iterNum)

        # confusion_mat, overall_acc, class_acc, avg_acc, kappa_score = result_reports(test_patch_hsi, test_patch_lidar, test_label, dataset, model, iterNum)
        KAPPA.append(kappa_score)
        OA.append(overall_acc)
        AA.append(avg_acc)
        ELEMENT_ACC[iterNum, :] = class_acc
        torch.save(model, dataset+'/best_model_'+FileName+'_Iter'+str(iterNum)+'.pt')
        print('\n')
        print("Overall Accuracy = ", overall_acc)
        print('\n')

    print(f"---------- Training Finished for {dataset} dataset -----------")
    print("\nThe Confusion Matrix")
    log_result(OA, AA, KAPPA, ELEMENT_ACC, './' + dataset + '/' + FileName + '_Report_' + dataset + '.txt')
