import h5py
import math
import numpy as np
import pickle
import torch

from skimage.measure import block_reduce

##########
# Sample #
##########

# There are two samples varying in hatch spacing and velocity
DATASETS = ["Spacing", "Velocity"]

# Frame width and height of pyrometry image in pixels.
PYROMETRY_X_PIXELS = 65
PYROMETRY_Y_PIXELS = 80

# X and Y of the cropped CT samples
CT_X_VOXELS = 423
CT_Y_VOXELS = 520
# VOXELS_X = 423
# VOXELS_Y = 520

####################
# Unit Conversions #
####################

PYROMETRY_X_Y_MICRONS = 23.6
VOXEL_X_Y_Z_MICRONS = 3.63

# Size of pyrometry frames converted to voxels.
PYROMETRY_X_VOXELS = round(PYROMETRY_X_PIXELS * PYROMETRY_X_Y_MICRONS / VOXEL_X_Y_Z_MICRONS)
PYROMETRY_Y_VOXELS = round(PYROMETRY_Y_PIXELS * PYROMETRY_X_Y_MICRONS / VOXEL_X_Y_Z_MICRONS)

###############
# Build Layer #
###############

# Total build layers, this is same for Spacing and Velocity.
BUILD_LAYERS = 159

# Build layer height or also distance between pyrometry layers.
BUILD_LAYER_MICRONS = 30 

# Converts build layer into voxel space
# (30 microns / 3.63 microns per voxel = 8.26 voxels)
BUILD_LAYER_VOXELS = BUILD_LAYER_MICRONS / VOXEL_X_Y_Z_MICRONS

# Offsets in [z, y, x] direction in voxels (provided by Andrew Polonsky).
BUILD_LAYER_OFFSETS = dict({
    "Spacing": [-18, 0, 63],
    "Velocity": [-14, -2, 73]
})

# Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, indexes = None, dev = False, verbose = False):
        self.config = config
        self.verbose = verbose

        self.dataset = config['dataset']
        self.dev = dev
        self.datasets_dir = f"{self.config['datasets_dir']}/{self.config['dataset']}"

        if (self.config['dataset'] == 'All'):
            self.datasets_dir = [
                f"{self.config['datasets_dir']}/Spacing",
                f"{self.config['datasets_dir']}/Velocity"
            ]

        self.indexes = indexes
        self.x_crop_voxel = round(
            self.config["x_crop_pixel"]
                * PYROMETRY_X_Y_MICRONS
                / VOXEL_X_Y_Z_MICRONS
        )
        self.y_crop_voxel = round(
            self.config["y_crop_pixel"]
                * PYROMETRY_X_Y_MICRONS
                / VOXEL_X_Y_Z_MICRONS
        )

        # Initialize
        self.pyrometry_map = []
        self.layer_frames = []
        self.load_pyrometry()

        self.sample = []
        self.segmented = []
        self.pores = []
        self.pore_ids = []
        self.equivalent_diameter_voxel = []
        self.equivalent_diameter_pore = []
        self.load_ct()

        # self.equivalent_diameter_pore_metrics = dict()
        # self.compile_metrics()


    def __getitem__(self, index):

        #########
        # Index #
        #########

        # dataset_index: 0 - Spacing, 1 - Velocity
        
        if isinstance(self.indexes, np.ndarray):
            # Uses specific layer index provided by pre-split train / test.
            # Otherwise `index` will be build layer index (i.e. 0 - 158)
            index = self.indexes[index]
            (dataset_index, index) = divmod(index, BUILD_LAYERS)
            if self.verbose:
                print(f"dataset_index: {dataset_index}, layer_index: {index}")
        else:
            print("TRAINING ON ENTIRE DATASET")

        # Retrieves list of frame files for specific build layer.
        frame_files = self.layer_frames[dataset_index][index]

        ##########
        # Inputs #
        ##########

        # Creates empty `frames` variable to store loaded frames.
        frames = []

        # Loads in the image data for each frame.
        # `[:self.frame_length]` limits longer videos to specified frame length.
        for frame_file in frame_files[:self.config["frame_length"]]:

            # Loads the specified frame from dataset directory and retrieves
            # temperature estimates, essentially 2D grayscale image.
            frame_data = open(
                f"{self.datasets_dir[dataset_index]}/beta=0.7_pickles/{frame_file}", "rb"
            )
            frame = pickle.load(frame_data)
            frame = np.array(frame["temperature_estimate"])

            # Crops loaded frame from 65px X 80px to 64px X 64px
            frame = frame[
                self.config["x_crop_pixel"]:-self.config["x_crop_pixel"],
                self.config["y_crop_pixel"]:
            ]

            # Adds cropped frame to frames list.
            frames.append(frame)


        # Pads frames to make consistent input shape of (`frame_length`, 64, 64)
        frames = np.pad(
            np.array(frames),
            pad_width=((0, self.config["frame_length"] - len(frames)), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0
        )


        # Stacks frames for single channel to n channels
        frames = np.stack((frames,)*self.config["channels"], axis=0)

        # Adds extra dimension for single no rotations (1, `frame_length`, 64, 64)
        # frames = np.expand_dims(frames, axis=0)

        ##########
        # Labels #
        ##########

        # Retrieves just the thresholded pore values, cropped to match thermals.
        ct_build_layers = self.get_build_layer(index, dataset_index)
        
        cts = []

        for ct in ct_build_layers:

            # Crops CT with the appropriate conversions from pixels
            ct = ct[
                :,
                self.x_crop_voxel:-self.x_crop_voxel, # Crop the left and right
                self.y_crop_voxel:                    # Only crop a bit off the top
            ]

            # Binarizes downsampled ct.
            if (self.config["ct_type"] != "pore_ids"):
                # Down samples heatmap in both the x and y direction but keeps z same.
                downsampling = (
                    1,
                    self.config["downsampling_factor"],
                    self.config["downsampling_factor"]
                )

                # Apply max pooling to downsample the array
                ct = block_reduce(ct, downsampling)
                ct = np.where(ct > 0, 1, 0)

            # Futher crops CT data to shape (9, 64, 64)
            ct = ct[
                :,
                self.config["x_crop_adjust_voxel"]:-self.config["x_crop_adjust_voxel"],
                self.config["y_crop_adjust_voxel"]:-self.config["y_crop_adjust_voxel"]
            ]

            ct = np.transpose(ct, (1, 2, 0))

            # Expands shape of label for frame dimension
            # ct = np.expand_dims(ct, axis=1)

            cts.append(ct)

        selected_ct = cts[4] # `layer_ct_pores_threshold`

        if (self.config["ct_type"] == "pores"):
            selected_ct = cts[3] # `layer_ct_pores`
        elif(self.config["ct_type"] == "segmented"):
            selected_ct = cts[1] # `layer_ct_segmented`
        elif(self.config["ct_type"] == "segmented_thresholded"):
            selected_ct = cts[2] # `layer_ct_segmented_thresholded`
        elif(self.config["ct_type"] == "sample"):
            selected_ct = cts[0] # `layer_ct_sample`
        elif(self.config["ct_type"] == "pore_ids"):
            selected_ct = cts[5] # `layer_ct_pore_ids
            pore_count = np.unique(selected_ct[selected_ct != 0])
            if (self.dev):
                return (torch.tensor(frames).float(), torch.tensor(len(pore_count)).float(), cts[5])
            return (torch.tensor(frames).float(), torch.tensor(len(pore_count)).float())

        ########
        # Item #
        ########

        if (self.dev):
            return (
                torch.tensor(frames).float(),
                torch.tensor(selected_ct).float(),
                cts
            )

        return (torch.tensor(frames).float(), torch.tensor(selected_ct).float())

    def __len__(self):
        """
        Returns number of layers within dataset
        """

        # Limits the iteration to that of the train / test split.
        if isinstance(self.indexes, np.ndarray):

            return len(self.indexes)

        if self.dataset == "All":
            return BUILD_LAYERS * 2
        return BUILD_LAYERS 

    def load_ct(self):
        """
        Loads in CT Data from `.h5` file.
        """

        # Returns reference for voxel data value
        # https://stackoverflow.com/a/41949774/10521456
        for dataset_index, dataset_dir in enumerate(self.datasets_dir):
            if dataset_index == 0:
                h5_file = h5py.File(f"{dataset_dir}/Spacing.h5")
            else:
                h5_file = h5py.File(f"{dataset_dir}/Velocity.h5")

            # Voxel Data
            self.sample.append(h5_file["CTDataContainer"]["VoxelData"]["Sample"])
            self.segmented.append(h5_file["CTDataContainer"]["VoxelData"]["Segmented"])
            self.pores.append(h5_file["CTDataContainer"]["VoxelData"]["Pores"])
            self.pore_ids.append(h5_file["CTDataContainer"]["VoxelData"]["PoreIds"])
            self.equivalent_diameter_voxel.append(h5_file["CTDataContainer"]["VoxelData"]["EqDiam_Vox"])

            # Pore Data
            self.equivalent_diameter_pore.append(h5_file["CTDataContainer"]["PoreData"]["EquivalentDiameters"])

    def load_pyrometry(self):
        """
        Loads centroid values and layer frames associated with pyrometry.
        """

        for dataset_dir in self.datasets_dir:
            # Opens dictionary of pyrometry such as frame files for each layer. 
            with open(f"{dataset_dir}/pyrometry_map.p", "rb") as map_file:
                pyrometry_map = pickle.load(map_file)
                self.pyrometry_map.append(pyrometry_map)

                # List of frame files associated with each one of the 159 build layers.
                self.layer_frames.append(pyrometry_map["layer_frames"])

    # def compile_metrics(self):
    #     """
    #     Determine Metrics for CT Pore data.
    #     """
    #     eq_diameter_pore = np.array(self.equivalent_diameter_pore).flatten()
    #     self.equivalent_diameter_pore_metrics["min"] = np.min(eq_diameter_pore)
    #     self.equivalent_diameter_pore_metrics["max"] = np.max(eq_diameter_pore)
    #     self.equivalent_diameter_pore_metrics["mean"] = np.mean(eq_diameter_pore)
    #     self.equivalent_diameter_pore_metrics["std"] = np.std(eq_diameter_pore)
    #     self.equivalent_diameter_pore_metrics["median"] = np.median(eq_diameter_pore)

    def get_build_layer(self, layer_index = 0, dataset_index = 0):
        """
        Crops CT data lengthwise to fit camera viewport and returns sample,
        segmented, and pores CT data of the build layer specific to layer index.
        """
        # Voxel Z Stop (index of the top of the layer)
        # Example for when layer index = 0
        # + 8.26 ~ 8 voxels per layer (hence +1 for pyrometry on top of CT)
        # Also allow for visual adjustment with `self.layer_index_offset`.
        build_layer_index = layer_index + 1 + self.config["layer_index_offset"]

        # + build layer index * voxels per layer (Initial ct voxel index)
        ct_voxel_index = math.floor(build_layer_index * BUILD_LAYER_VOXELS)

        # + 18 or 14 voxel offset in Z direction (shift ct in voxel space)
        dataset = "Spacing"
        if dataset_index == 1:
            dataset = "Velocity"
        ct_z_stop_index = ct_voxel_index + abs(BUILD_LAYER_OFFSETS[dataset][0])
        # = 58 or 54 for voxel index

        # Voxel Z Start (index of the bottom of the layer)
        # How deep the bottom layer goes, with a minimum of 0 (bottom index).
        ct_z_start_index = max(0, ct_z_stop_index - self.config["voxels_below_layer"])

        # Voxel Y Start (index of the back of the sample)
        ct_y_start_index = 0

        # Voxel Y Stop (index of the front of the sample)
        ct_y_stop_index = PYROMETRY_Y_VOXELS 

        # Voxel X Start (index of the left of the sample)
        ct_x_start_index = 0 

        # Voxel X Stop (index of the right of the sample)
        ct_x_stop_index = PYROMETRY_X_VOXELS

        layer_ct_sample = self.sample[dataset_index][
            ct_z_start_index:ct_z_stop_index,
            ct_y_start_index:ct_y_stop_index,
            ct_x_start_index:ct_x_stop_index,
        ]

        layer_ct_segmented = self.segmented[dataset_index][
            ct_z_start_index:ct_z_stop_index,
            ct_y_start_index:ct_y_stop_index,
            ct_x_start_index:ct_x_stop_index,
        ]

        layer_ct_pores = self.pores[dataset_index][
            ct_z_start_index:ct_z_stop_index,
            ct_y_start_index:ct_y_stop_index,
            ct_x_start_index:ct_x_stop_index,
        ]

        layer_ct_pore_ids = self.pore_ids[dataset_index][
            ct_z_start_index:ct_z_stop_index,
            ct_y_start_index:ct_y_stop_index,
            ct_x_start_index:ct_x_stop_index,
        ]

        layer_ct_equivalent_diameter_voxel = self.equivalent_diameter_voxel[dataset_index][
            ct_z_start_index:ct_z_stop_index,
            ct_y_start_index:ct_y_stop_index,
            ct_x_start_index:ct_x_stop_index,
        ]

        # mean = self.equivalent_diameter_pore_metrics["mean"]
        # std = self.equivalent_diameter_pore_metrics["std"]

        # threshold = mean + (self.config["std_threshold_eq_diameter_pores"] * std)

        # layer_ct_equivalent_diameter_voxel_threshold = np.where(layer_ct_equivalent_diameter_voxel.squeeze() < threshold, 0, 1) 

        # binary_pores = np.where(layer_ct_pores.squeeze() > 0, 1, 0)

        # # Subtracts Thresholded pores from all pores to get negative mapping.
        # below_threshold = binary_pores - layer_ct_equivalent_diameter_voxel_threshold

        # layer_ct_segmented_thresholded = np.where(below_threshold == 1, 255, layer_ct_segmented.squeeze())
        # layer_ct_segmented_thresholded = np.expand_dims(layer_ct_segmented_thresholded, axis=3)

        # # layer_ct_pores_thresholded = np.expand_dims(layer_ct_equivalent_diameter_voxel_threshold, axis=3)
        # layer_ct_pores_thresholded = layer_ct_equivalent_diameter_voxel_threshold

        return (
            layer_ct_sample.squeeze(),
            layer_ct_segmented.squeeze(),
            layer_ct_segmented.squeeze(),
            # layer_ct_segmented_thresholded.squeeze(),
            layer_ct_pores.squeeze(),
            layer_ct_pores.squeeze(),
            # layer_ct_pores_thresholded,
            layer_ct_pore_ids.squeeze(),
        )

# Splits dataset indexes to train and test within sample steps.
SEED = 0

def split_dataset(train_fraction, test_fraction, verbose = False, dataset = "Spacing"):
  all_length = BUILD_LAYERS
  steps = 10
  if dataset == "All":
    if verbose: print(f"dataset for split: {dataset}")
    all_length = BUILD_LAYERS * 2
    steps = steps * 2

  train_indexes = np.array([])
  test_indexes = np.array([])

  np.random.seed(SEED)

  layers_per_step = 16

  train_size = int(layers_per_step * train_fraction)
  test_size = int(layers_per_step * test_fraction)

  for step_index in range(steps):
      start_index = step_index * layers_per_step
      end_index = min((step_index + 1) * layers_per_step, all_length)
      layer_indexes = np.arange(start_index,end_index)

      np.random.shuffle(layer_indexes)

      train_indexes = np.concatenate([train_indexes, layer_indexes[:train_size]])
      test_indexes = np.concatenate([test_indexes, layer_indexes[train_size:train_size + test_size:]])

  np.random.shuffle(train_indexes)
  np.random.shuffle(test_indexes)

  # Converts to int datatype.
  train_indexes = np.array(train_indexes, dtype=int)
  test_indexes = np.array(test_indexes, dtype=int)

  if verbose:
      print(f"train: {train_fraction}, test: {test_fraction}")
      print(f"train_indexes ({len(train_indexes)}): {train_indexes}")
      print(f"test_indexes ({len(test_indexes)}): {test_indexes}")

  return (train_indexes, test_indexes)