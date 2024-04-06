import h5py
import math
import numpy as np
import pickle
import torch
import torch.nn.functional as F

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, indexes = None, dev = False):
        self.config = config
        self.dev = dev
        self.indexes = indexes

        self.datasets_dir = f"{self.config['datasets_dir']}/{self.config['dataset']}"

        # Initialize
        self.pore_boundaries = np.load(f"{self.datasets_dir}/pore_boundaries.npy")

        self.pore_centroids = None
        self.pore_ids = None
        self.load_ct()

    def __getitem__(self, index):
        #########
        # Index #
        #########
        
        if isinstance(self.indexes, np.ndarray):
            index = self.indexes[index]
        else:
            print("TRAINING ON ENTIRE DATASET")

        pore_id = index + 1
        pore_boundary = self.pore_boundaries[index]

        z_min = pore_boundary[0][0]
        z_max = pore_boundary[0][1]
        y_min = pore_boundary[1][0]
        y_max = pore_boundary[1][1]
        x_min = pore_boundary[2][0]
        x_max = pore_boundary[2][1]

        pore_id_slice = self.pore_ids[z_min:z_max,y_min:y_max,x_min:x_max]

        # Only considers pores that match the pore id.
        pore_id_slice[pore_id_slice != pore_id] = 0

        # pore_tensor = torch.tensor(pore_id_slice.astype(np.uint8))
        pore_tensor = torch.tensor(pore_id_slice.astype(np.uint8)).float()

        bounds = self.config["pore_boundary_size"]

        # Remaining area of tensor not covered in overlap 
        spaces = [bounds[i] - pore_id_slice.shape[i] for i in range(len(bounds))]

        pad_sizes = []
        for space in spaces:
            quotient, remainder = divmod(space, 2)
            side_1_padding = quotient
            pad_sizes.append(side_1_padding)
            side_2_padding = quotient + remainder
            pad_sizes.append(side_2_padding)

        # Makes sure that that pad sizes are in x y z input format.
        pad_sizes.reverse()
        # pad_sizes.append(0)
        # pad_sizes.append(0)

        padded_pore_tensor = F.pad(pore_tensor, pad_sizes)

        # Boolean and uint8 use the same amount of memory.
        # https://github.com/pytorch/pytorch/issues/41571
        # pore_tensor = torch.tensor(pore_id_slice.astype(np.uint8))
        # print(f"padded_pore_tensor.shape: {padded_pore_tensor.shape}")

        return (padded_pore_tensor, self.pore_centroids[index])

    def __len__(self):
        """
        Returns number of segmented pores within dataset
        """

        # Limits the iteration to that of the train / test split.
        if isinstance(self.indexes, np.ndarray):

            return len(self.indexes)

        return len(self.pore_boundaries)


    def load_ct(self):
        """
        Loads in CT Data from `.h5` file.
        """

        # Returns reference for voxel data value
        # https://stackoverflow.com/a/41949774/10521456
        h5_file = h5py.File(f"{self.datasets_dir}/{self.config['dataset']}.h5")

        # Voxel Data
        self.pore_ids = np.array(h5_file["CTDataContainer"]["VoxelData"]["PoreIds"])

        # Removes exta dimension around values
        self.pore_ids = self.pore_ids.squeeze()

        # Pore Data
        self.pore_centroids = np.array(h5_file["CTDataContainer"]["PoreData"]["Centroids"])

SEED = 0

def split_dataset(train_fraction = 0.8, test_fraction = 0.2, verbose = False):
    all_length = 2068 # Spacing

    train_indexes = np.array([])
    test_indexes = np.array([])

    np.random.seed(SEED)

    # Generating random indexes for training and testing data
    all_indexes = np.arange(all_length)
    np.random.shuffle(all_indexes)

    train_size = int(train_fraction * all_length)
    train_indexes = all_indexes[:train_size]
    test_indexes = all_indexes[train_size:]

    # Converts to int datatype.
    train_indexes = np.array(train_indexes, dtype=int)
    test_indexes = np.array(test_indexes, dtype=int)

    if verbose:
        print(f"train: {train_fraction}, test: {test_fraction}")
        print(f"train_indexes ({len(train_indexes)}): {train_indexes}")
        print(f"test_indexes ({len(test_indexes)}): {test_indexes}")

    return (train_indexes, test_indexes)