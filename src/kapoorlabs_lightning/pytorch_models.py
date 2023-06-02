import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_functions import get_graph_feature, knn, local_cov, local_maxpool


class ClusteringLayer(nn.Module):
    def __init__(self, num_features=10, num_clusters=10, alpha=1.0):
        super().__init__()
        self.num_features = num_features
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.weight = nn.Parameter(
            torch.Tensor(self.num_clusters, self.num_features)
        )
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(1) - self.weight
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha + 1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x

    def extra_repr(self):
        return "in_features={}, out_features={}, alpha={}".format(
            self.num_features, self.num_clusters, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)


class CloudAutoEncoder(nn.Module):
    def __init__(
        self,
        num_features,
        k=20,
        encoder_type="dgcnn",
        decoder_type="foldingnet",
        shape="plane",
        sphere_path="./sphere.npy",
        gaussian_path="./gaussian.npy",
        std=0.3,
    ):
        super().__init__()
        self.k = k
        self.num_features = num_features
        self.shape = shape
        self.sphere_path = sphere_path
        self.gaussian_path = gaussian_path
        self.std = std

        assert encoder_type.lower() in [
            "foldingnet",
            "dgcnn",
            "dgcnn_orig",
        ], "Please select an encoder type from either foldingnet or dgcnn."

        assert decoder_type.lower() in [
            "foldingnet",
            "foldingnetbasic",
        ], "Please select an decoder type from either foldingnet."

        self.encoder_type = encoder_type.lower()
        self.decoder_type = decoder_type.lower()
        if self.encoder_type == "dgcnn":
            self.encoder = DGCNNEncoder(
                num_features=self.num_features, k=self.k
            )
        # elif self.encoder_type == "dgcnn_orig":
        #     self.encoder = DGCNN(num_features=self.num_features, k=self.k)
        else:
            self.encoder = FoldNetEncoder(
                num_features=self.num_features, k=self.k
            )

        if self.decoder_type == "foldingnet":
            self.decoder = FoldNetDecoder(
                num_features=self.num_features,
                shape=self.shape,
                sphere_path=self.sphere_path,
                gaussian_path=self.gaussian_path,
                std=self.std,
            )
        else:
            self.decoder = FoldingNetBasicDecoder(
                num_features=self.num_features,
                shape=self.shape,
                sphere_path=self.sphere_path,
                gaussian_path=self.gaussian_path,
                std=self.std,
            )

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(x=features)
        return output, features


class DeepEmbeddedClustering(nn.Module):
    def __init__(self, encoder, decoder, num_clusters):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_clusters = num_clusters
        self.clustering_layer = ClusteringLayer(
            num_features=self.encoder.num_features,
            num_clusters=self.num_clusters,
        )

    def forward(self, x):
        features = self.encoder(x)
        clusters = self.clustering_layer(features)
        output = self.decoder(features)
        return output, features, clusters


class FoldingModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.folding1 = nn.Sequential(
            nn.Linear(512 + 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        self.folding2 = nn.Sequential(
            nn.Linear(512 + 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

    def forward(self, x, grid):
        cw_exp = x.expand(-1, grid.shape[1], -1)

        cat1 = torch.cat((cw_exp, grid), dim=2)
        folding_result1 = self.folding1(cat1)
        cat2 = torch.cat((cw_exp, folding_result1), dim=2)
        folding_result2 = self.folding2(cat2)
        return folding_result2


class FoldingNetBasicDecoder(nn.Module):
    def __init__(
        self,
        num_features,
        shape="plane",
        sphere_path="./sphere.npy",
        gaussian_path="./gaussian.npy",
        std=0.3,
    ):
        super().__init__()

        # initialise deembedding
        self.lin_features_len = 512
        self.num_features = num_features
        if self.num_features < self.lin_features_len:
            self.deembedding = nn.Linear(
                self.num_features, self.lin_features_len, bias=False
            )

        if shape == "plane":
            # make grid
            range_x = torch.linspace(-std, std, 45)
            range_y = torch.linspace(-std, std, 45)
            x_coor, y_coor = torch.meshgrid(range_x, range_y, indexing="ij")
            self.grid = (
                torch.stack([x_coor, y_coor], axis=-1).float().reshape(-1, 2)
            )
        elif shape == "sphere":
            self.grid = torch.tensor(np.load(sphere_path))
        elif self.shape == "gaussian":
            self.grid = torch.tensor(np.load(gaussian_path))

        # initialise folding module
        self.folding = FoldingModule()

    def forward(self, x):
        if self.num_features < self.lin_features_len:
            x = self.deembedding(x)
            x = x.unsqueeze(1)

        else:
            x = x.unsqueeze(1)
        grid = self.grid.cuda().unsqueeze(0).expand(x.shape[0], -1, -1)
        outputs = self.folding(x, grid)
        return outputs


class FoldNetDecoder(nn.Module):
    def __init__(
        self,
        num_features,
        shape="plane",
        sphere_path="./sphere.npy",
        gaussian_path="./gaussian.npy",
        std=0.3,
    ):
        super().__init__()
        self.m = 2025  # 45 * 45.
        self.std = std
        self.meshgrid = [[-std, std, 45], [-std, std, 45]]
        self.shape = shape
        if shape == "sphere":
            self.sphere = np.load(sphere_path)
        if shape == "gaussian":
            self.gaussian = np.load(gaussian_path)
        self.num_features = num_features
        if self.shape == "plane":
            self.folding1 = nn.Sequential(
                nn.Conv1d(512 + 2, 512, 1),
                nn.ReLU(),
                nn.Conv1d(512, 512, 1),
                nn.ReLU(),
                nn.Conv1d(512, 3, 1),
            )
        else:
            self.folding1 = nn.Sequential(
                nn.Conv1d(512 + 3, 512, 1),
                nn.ReLU(),
                nn.Conv1d(512, 512, 1),
                nn.ReLU(),
                nn.Conv1d(512, 3, 1),
            )

        self.folding2 = nn.Sequential(
            nn.Conv1d(512 + 3, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1),
        )

        self.lin_features_len = 512
        if self.num_features < self.lin_features_len:
            self.deembedding = nn.Linear(
                self.num_features, self.lin_features_len, bias=False
            )

    def build_grid(self, batch_size):
        if self.shape == "plane":
            x = np.linspace(*self.meshgrid[0])
            y = np.linspace(*self.meshgrid[1])
            points = np.array(list(itertools.product(x, y)))
        elif self.shape == "sphere":
            points = self.sphere
        elif self.shape == "gaussian":
            points = self.gaussian

        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def forward(self, x):
        if self.num_features < self.lin_features_len:
            x = self.deembedding(x)
            x = x.unsqueeze(1)

        else:
            x = x.unsqueeze(1)

        x = x.transpose(1, 2).repeat(1, 1, self.m)
        points = self.build_grid(x.shape[0]).transpose(1, 2)
        if x.get_device() != -1:
            points = points.cuda(x.get_device())
        cat1 = torch.cat((x, points), dim=1)

        folding_result1 = self.folding1(cat1)
        cat2 = torch.cat((x, folding_result1), dim=1)
        folding_result2 = self.folding2(cat2)
        output = folding_result2.transpose(1, 2)
        return output


class DGCNNEncoder(nn.Module):
    def __init__(self, num_features, k=20):
        super().__init__()
        self.k = k
        self.num_features = num_features

        self.conv1 = nn.Sequential(
            nn.Conv2d(3 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.clustering = None
        self.lin_features_len = 512
        if (self.num_features < self.lin_features_len) or (
            self.num_features > self.lin_features_len
        ):
            self.flatten = Flatten()
            self.embedding = nn.Linear(
                self.lin_features_len, self.num_features, bias=False
            )

    def forward(self, x):
        x = x.transpose(2, 1)

        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x0 = self.conv5(x)
        x = x0.max(dim=-1, keepdim=False)[0]
        feat = x.unsqueeze(1)

        if (self.num_features < self.lin_features_len) or (
            self.num_features > self.lin_features_len
        ):
            x = self.flatten(feat)
            features = self.embedding(x)
        else:
            features = torch.reshape(torch.squeeze(feat), (batch_size, 512))

        return features


class FoldNetEncoder(nn.Module):
    def __init__(self, num_features, k):
        super().__init__()
        if k is None:
            self.k = 16
        else:
            self.k = k
        self.n = 2048
        self.num_features = num_features
        self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
        )
        self.clustering = None
        self.lin_features_len = 512
        if (self.num_features < self.lin_features_len) or (
            self.num_features > self.lin_features_len
        ):
            self.embedding = nn.Linear(
                self.lin_features_len, self.num_features, bias=False
            )

    def graph_layer(self, x, idx):
        x = local_maxpool(x, idx)
        x = self.linear1(x)
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = local_maxpool(x, idx)
        x = self.linear2(x)
        x = x.transpose(2, 1)
        x = self.conv2(x)
        return x

    def forward(self, pts):
        pts = pts.transpose(2, 1)
        batch_size = pts.size(0)
        idx = knn(pts, k=self.k)
        x = local_cov(pts, idx)
        x = self.mlp1(x)
        x = self.graph_layer(x, idx)
        x = torch.max(x, 2, keepdim=True)[0]
        x = self.mlp2(x)
        feat = x.transpose(2, 1)
        if (self.num_features < self.lin_features_len) or (
            self.num_features > self.lin_features_len
        ):
            x = self.flatten(feat)
            features = self.embedding(x)
        else:
            features = torch.reshape(torch.squeeze(feat), (batch_size, 512))

        return features


class DGCNN(nn.Module):
    def __init__(self, num_features=2, k=20, emb_dims=512):
        super().__init__()
        self.num_features = num_features
        self.k = k
        self.emb_dims = emb_dims
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)
        self.dropout = 0.2

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.linear1 = nn.Linear(self.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout)
        self.linear3 = nn.Linear(256, num_features)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        """
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0)
        number of batches,
        will flatten to be 1 * nb_elements.
        """
        batch_size = input.size(0)
        out = input.view(batch_size, -1)
        return out


__all__ = [
    "CloudAutoEncoder",
    "DGCNNEncoder",
    "FoldNetEncoder",
    "FoldNetDecoder",
    "FoldingNetBasicDecoder",
    "Flatten",
    "FoldingModule",
    "DGCNN",
    "DeepEmbeddedClustering",
    "ClusteringLayer",
]
