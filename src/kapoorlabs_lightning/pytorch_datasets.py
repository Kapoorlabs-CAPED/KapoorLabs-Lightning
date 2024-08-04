import json
import os
from glob import glob
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
from pyntcloud import PyntCloud
from sklearn import preprocessing
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
import networkx as nx


SHAPE_FEATURES = [
    "Radius",
    "Eccentricity_Comp_First",
    "Eccentricity_Comp_Second",
    "Eccentricity_Comp_Third",
    "Local_Cell_Density",
    "Surface_Area",
]

DYNAMIC_FEATURES = [
    "Speed",
    "Motion_Angle_Z",
    "Motion_Angle_Y",
    "Motion_Angle_X",
    "Acceleration",
    "Distance_Cell_mask",
    "Radial_Angle_Z",
    "Radial_Angle_Y",
    "Radial_Angle_X",
    "Cell_Axis_Z",
    "Cell_Axis_Y",
    "Cell_Axis_X",
]


class TrackingDataset(Dataset):
    def __init__(self, tracks_dataframe: pd.DataFrame):
        self.tracks_dataframe = tracks_dataframe.copy()
        self.unique_trackmate_track_ids = self.tracks_dataframe[
            "TrackMate Track ID"
        ].unique()
        parent_dict = {}
        t_min_dict = {}
        t_max_dict = {}
        for trackmate_track_id in self.unique_trackmate_track_ids:
            subset = self.tracks_dataframe[
                (self.tracks_dataframe["TrackMate Track ID"] == trackmate_track_id)
            ].sort_values(by="t")
            sorted_subset = sorted(subset["Track ID"].unique())

            for tracklet_id in sorted_subset:

                tracklets_dataframe = tracks_dataframe[
                    (tracks_dataframe["Track ID"] == tracklet_id)
                ].sort_values(by="t")

                if len(sorted_subset) == 1:
                    parent_dict[tracklet_id] = 0
                else:
                    parent_dict[tracklet_id] = sorted_subset[0]

                t_min_dict[tracklet_id] = tracklets_dataframe["t"].min()
                t_max_dict[tracklet_id] = tracklets_dataframe["t"].max()
                coords = tracklets_dataframe[["z", "y", "x"]].values
                timepoints = tracklets_dataframe[["t"]].values
                shape_featues = tracklets_dataframe[SHAPE_FEATURES]
                dynamic_features = tracklets_dataframe[DYNAMIC_FEATURES]
                print(coords, timepoints, shape_featues, dynamic_features)
        self.tracks_dataframe["Parent"] = self.tracks_dataframe["Track ID"].map(
            parent_dict
        )
        self.tracks_dataframe["t1"] = self.tracks_dataframe["Track ID"].map(t_min_dict)
        self.tracks_dataframe["t2"] = self.tracks_dataframe["Track ID"].map(t_max_dict)
        self._convert_to_ctc_dataframe()
        self._ctc_lineages()

    def _convert_to_ctc_dataframe(self):

        self.ctc_tracks_dataframe = self.tracks_dataframe[
            ["Track ID", "t1", "t2", "Parent"]
        ].astype("int")
        self.ctc_tracks_dataframe.rename(columns={"Track ID": "Label"}, inplace=True)
        self.ctc_tracks_dataframe.drop_duplicates(inplace=True)

    def _ctc_lineages(self):
        self.graph = nx.DiGraph()
        for _, row in self.ctc_tracks_dataframe.iterrows():
            label = row["Label"]
            parent_id = row["Parent"]
            self.graph.add_node(label)
            if parent_id != 0:
                self.graph.add_edge(parent_id, label)

    def _ctc_assoc_matrix(self):

        matched_gt = self.ctc_tracks_dataframe["Label"].unique()
        num_labels = len(matched_gt)
        fwd_map = {label: idx for idx, label in enumerate(matched_gt)}
        self.association_matrix = np.zeros((num_labels, num_labels), dtype=bool)

        for _, row in self.ctc_tracks_dataframe.iterrows():
            gt_tracklet_id = row["Label"]
            ancestors = []
            descendants = []
            for n in nx.descendants(self.graph, gt_tracklet_id):
                if n in fwd_map:
                    descendants.append(fwd_map[n])
            for n in nx.ancestors(self.graph, gt_tracklet_id):
                if n in fwd_map:
                    ancestors.append(fwd_map[n])

            self.association_matrix[
                fwd_map[gt_tracklet_id],
                np.array([fwd_map[gt_tracklet_id], *ancestors, *descendants]),
            ] = True

    def __len__(self):
        return len(self.unique_trackmate_track_ids)


class MitosisDataset(Dataset):
    def __init__(self, arrays, labels):
        self.arrays = arrays
        self.labels = labels
        self.input_channels = arrays.shape[2]

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        array = self.arrays[idx]
        array = torch.tensor(array).permute(1, 0).float()
        label = torch.tensor(self.labels[idx])

        return array, label


class H5MitosisDataset(Dataset):
    def __init__(self, h5_file, data_key, label_key):
        self.h5_file = h5_file
        self.data_key = data_key
        self.label_key = label_key
        self.data_label = h5py.File(self.h5_file, "r")
        self.data = self.data_label[data_key]
        self.targets = self.data_label[label_key]
        self.input_channels = np.asarray(self.data[0]).shape[1]
  
    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx):
       
            array = torch.from_numpy(np.asarray(self.data[idx])).permute(1, 0).float()

            label = torch.from_numpy(np.asarray(self.targets[idx]))

            return array, label


class PointCloudDataset(Dataset):
    def __init__(self, points_dir, centre=True, scale_z=1.0, scale_xy=1.0):
        self.points_dir = points_dir
        self.centre = centre
        self.scale_z = scale_z
        self.scale_xy = scale_xy
        self.p = Path(self.points_dir)
        self.files = list(self.p.glob("**/*.ply"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # read the image
        file = self.files[idx]
        point_cloud = PyntCloud.from_file(str(file))
        mean = 0
        point_cloud = torch.tensor(point_cloud.points.values)
        if self.centre:
            mean = torch.mean(point_cloud, 0)

        scale = torch.tensor([[self.scale_z, self.scale_xy, self.scale_xy]])
        point_cloud = (point_cloud - mean) / scale

        return point_cloud


class PointCloudNpzDataset(Dataset):
    def __init__(self, npz_path, centre=True, scale=1.0):
        self.npz_path = npz_path
        self.centre = centre
        self.scale = scale
        self.npzdata = np.load(npz_path)
        self.mesh_data = self.npzdata["mesh"].tolist()
        self.points_data = self.npzdata["points"].tolist()

    def __len__(self):
        return len(self.points_data)

    def __getitem__(self, idx):
        # read the image
        points = self.points_data[idx]

        point_cloud = PyntCloud(points)
        mean = 0
        point_cloud = torch.tensor(point_cloud.points.values)
        if self.centre:
            mean = torch.mean(point_cloud, 0)

        scale = torch.tensor([[self.scale, self.scale, self.scale]])
        point_cloud = (point_cloud - mean) / scale

        return point_cloud


class SingleCellDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        points_dir,
        img_size=400,
        transform=None,
        cell_component="cell",
        num_points=2048,
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = points_dir
        self.img_size = img_size
        self.transform = transform
        self.cell_component = cell_component
        self.num_points = num_points

        self.new_df = self.annot_df[
            (self.annot_df.xDim <= self.img_size)
            & (self.annot_df.yDim <= self.img_size)
            & (self.annot_df.zDim <= self.img_size)
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, "Treatment"]
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        if self.num_points == 4096:
            num_str = "_4096"
        elif self.num_points == 1024:
            num_str = "_1024"
        else:
            num_str = ""

        if self.cell_component == "cell":
            component_path = "stacked_pointcloud" + num_str
        else:
            component_path = "stacked_pointcloud_nucleus" + num_str

        img_path = os.path.join(
            self.img_dir,
            plate_num,
            component_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )
        image = PyntCloud.from_file(img_path + ".ply")
        image = torch.tensor(image.points.values)
        mean = torch.mean(image, 0)
        std = torch.tensor([[20.0, 20.0, 20.0]])
        image = (image - mean) / std

        # return the classical features as torch tensor
        feats = self.new_df.iloc[idx, 16:-4]
        feats = torch.tensor(feats)

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, treatment, feats, serial_number


class GefGapDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=100,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        cell_component="cell",
        norm_std=False,
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.norm_std = norm_std

        self.new_df = self.annot_df[
            (self.annot_df.xDim_cell <= self.img_size)
            & (self.annot_df.yDim_cell <= self.img_size)
            & (self.annot_df.zDim_cell <= self.img_size)
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        plate_num = self.new_df.loc[idx, "PlateNumber"]
        treatment = self.new_df.loc[idx, "GEF_GAP_GTPase"]
        plate = "Plate" + str(plate_num)
        if self.cell_component == "cell":
            component_path = "stacked_pointcloud"
            img_path = os.path.join(
                self.img_dir,
                plate,
                component_path,
                self.new_df.loc[idx, "serialNumber"],
            )
        else:
            component_path = "stacked_pointcloud_nucleus"
            img_path = os.path.join(
                self.img_dir,
                plate,
                component_path,
                "Cells",
                self.new_df.loc[idx, "serialNumber"],
            )

        image = PyntCloud.from_file(img_path + ".ply")
        image = image.points.values

        image = torch.tensor(image)
        mean = torch.mean(image, 0)
        if self.norm_std:
            std = torch.tensor([[20.0, 20.0, 20.0]])
        else:
            std = torch.abs(image - mean).max() * 0.9999999

        image = (image - mean) / std

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, treatment, serial_number


class ModelNet40(Dataset):
    def __init__(self, img_dir, train="train", transform=None):
        self.img_dir = Path(img_dir)
        self.train = train
        self.transform = transform
        self.files = list(self.img_dir.glob(f"**/{train}/*.ply"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # read the image
        file = self.files[idx]
        image = PyntCloud.from_file(str(file))
        image = torch.tensor(image.points.values)
        label = str(file.name)[:-9]
        image = (image - torch.mean(image, 0)) / (image.max())

        return image, label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype(
        "float32"
    )
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.choice(24) / 24
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(
        rotation_matrix
    )  # random rotation (x,z)
    return pointcloud


# The following was adapted from
# https://github.com/antao97/UnsupervisedPointCloudReconstruction
class ShapeNetDataset(Dataset):
    def __init__(
        self,
        root,
        dataset_name="modelnet40",
        num_points=2048,
        split="train",
        load_name=False,
        random_rotate=False,
        random_jitter=False,
        random_translate=False,
    ):
        assert dataset_name.lower() in [
            "shapenetcorev2",
            "shapenetpart",
            "modelnet10",
            "modelnet40",
        ]
        assert num_points <= 2048

        if dataset_name in ["shapenetpart", "shapenetcorev2"]:
            assert split.lower() in ["train", "test", "val", "trainval", "all"]
        else:
            assert split.lower() in ["train", "test", "all"]

        self.root = os.path.join(root, dataset_name + "*hdf5_2048")
        self.dataset_name = dataset_name
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate

        self.path_h5py_all = []
        self.path_json_all = []
        if self.split in ["train", "trainval", "all"]:
            self.get_path("train")
        if self.dataset_name in ["shapenetpart", "shapenetcorev2"]:
            if self.split in ["val", "trainval", "all"]:
                self.get_path("val")
        if self.split in ["test", "all"]:
            self.get_path("test")

        self.path_h5py_all.sort()
        data, label = self.load_h5py(self.path_h5py_all)
        if self.load_name:
            self.path_json_all.sort()
            self.name = self.load_json(self.path_json_all)  # load label name

        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0)

    def get_path(self, type):
        path_h5py = os.path.join(self.root, "*%s*.h5" % type)
        self.path_h5py_all += glob(path_h5py)
        if self.load_name:
            path_json = os.path.join(self.root, "%s*_id2name.json" % type)
            self.path_json_all += glob(path_json)
        return

    def load_h5py(self, path):
        all_data = []
        all_label = []
        for h5_name in path:
            f = h5py.File(h5_name, "r+")
            data = f["data"][:].astype("float32")
            label = f["label"][:].astype("int64")
            f.close()
            all_data.append(data)
            all_label.append(label)
        return all_data, all_label

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j = open(json_name, "r+")
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][: self.num_points]
        label = self.label[item]
        if self.load_name:
            name = self.name[item]  # get label name

        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)

        if self.load_name:
            return point_set, label, name
        else:
            return point_set, label

    def __len__(self):
        return self.data.shape[0]


class OPMDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=100,
        label_col="Treatment",
        transform=None,
        target_transform=None,
        cell_component="cell",
        norm_std=True,
        single_path="./",
        gef_path="./",
    ):
        self.annot_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.cell_component = cell_component
        self.norm_std = norm_std
        self.single_path = single_path
        self.gef_path = gef_path

        self.new_df = self.annot_df[
            (self.annot_df.xDim <= self.img_size)
            & (self.annot_df.yDim <= self.img_size)
            & (self.annot_df.zDim <= self.img_size)
            & (self.annot_df.Proximal == 1)
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        plate_num = self.new_df.loc[idx, "PlateNumber"]
        treatment = self.new_df.loc[idx, "Treatment"]
        plate = "Plate" + str(plate_num)

        if "accelerator" in self.new_df.loc[idx, "serialNumber"]:
            dat_type_path = self.single_path
            if self.cell_component == "cell":
                component_path = "stacked_pointcloud"
            elif self.cell_component == "smooth":
                component_path = "stacked_pointcloud_smoothed"
            else:
                component_path = "stacked_pointcloud_nucleus"

            img_path = os.path.join(
                self.img_dir,
                self.single_path,
                plate,
                component_path,
                treatment,
                str(self.new_df.loc[idx, "serialNumber"]),
            )

        else:
            dat_type_path = self.gef_path
            if self.cell_component == "cell":
                component_path = "stacked_pointcloud"
                img_path = os.path.join(
                    self.img_dir,
                    dat_type_path,
                    plate,
                    component_path,
                    str(self.new_df.loc[idx, "serialNumber"]),
                )
            else:
                component_path = "stacked_pointcloud_nucleus"
                img_path = os.path.join(
                    self.img_dir,
                    dat_type_path,
                    plate,
                    component_path,
                    "Cells",
                    str(self.new_df.loc[idx, "serialNumber"]),
                )

        image = PyntCloud.from_file(img_path + ".ply")
        image = image.points.values

        image = torch.tensor(image)
        mean = torch.mean(image, 0)
        if self.norm_std:
            std = torch.tensor([[20.0, 20.0, 20.0]])
        else:
            std = torch.std(image, 0)

        image = (image - mean) / std
        pc = PCA(n_components=3)
        u = torch.tensor(pc.fit_transform(image.numpy()))

        serial_number = self.new_df.loc[idx, "serialNumber"]

        return image, treatment, u, serial_number


class VesselMNIST3D(Dataset):
    def __init__(self, points_dir, centre=True, scale=1.0, partition="train"):
        self.points_dir = points_dir
        self.centre = centre
        self.scale = scale
        self.p = Path(self.points_dir)
        self.partition = partition
        self.path = self.p / partition
        self.files = list(self.path.glob("**/*.ply"))
        self.classes = [
            x.parents[0].name.replace("_pointcloud", "") for x in self.files
        ]

        self.le = preprocessing.LabelEncoder()
        self.class_labels = self.le.fit_transform(self.classes)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # read the image
        file = self.files[idx]
        label = self.class_labels[idx]
        class_name = self.classes[idx]
        point_cloud = PyntCloud.from_file(str(file))
        mean = 0
        point_cloud = torch.tensor(point_cloud.points.values)
        if self.centre:
            mean = torch.mean(point_cloud, 0)

        scale = torch.tensor([[self.scale, self.scale, self.scale]])
        point_cloud = (point_cloud - mean) / scale
        pc = PCA(n_components=3)
        u = torch.tensor(pc.fit_transform(point_cloud.numpy()))

        return (
            point_cloud,
            torch.tensor(label, dtype=torch.int64),
            u,
            class_name,
        )
