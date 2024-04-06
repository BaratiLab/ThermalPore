import argparse
import numpy as np
import os
import shutil
import wandb

from dataset import Dataset, split_dataset
from einops import rearrange
from initialize import initialize_model, load_config, write_config
from pprint import pprint
from torchinfo import summary
from train import train

run_dir = os.path.dirname(os.path.abspath(__file__))

#############
# Arguments #
#############

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="'cnn', 'vivit_dense', 'resnet'")
parser.add_argument("name", type=str, help="Name of run")

parser.add_argument(
    "--batch_size",
    type = int,
    default = 2,
    help = "Batch size of train and test",
    required = False
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="i.e. 'cuda:0', etc.",
    required=False
)
parser.add_argument(
    "--dataset",
    type=str,
    default="Spacing",
    help="Either 'Spacing' or 'Velocity'",
    required=False
)
parser.add_argument(
    "--epochs",
    type=int,
    default = 1000 + 1, # Outputs results from last epoch in same format.
    help="Number of epoches to run",
    required=False
)
parser.add_argument(
    "--epoch_write",
    type=int,
    default = 50,
    help="Number of epoches to run before writing",
    required=False
)
parser.add_argument(
    "--checkpoint_write",
    type=int,
    default = 250,
    help="Number of epoches to run before writing checkpoint",
    required=False
)
parser.add_argument(
    "--learning_rate",
    type = int,
    default = 1E-4,
    help = "i.e. '1E-4', '1E-5'",
    required = False
)
parser.add_argument(
    "--verbose",
    type = bool,
    default = False,
    help = "Runs print statements",
    required = False
)
parser.add_argument(
    "--wandb",
    type = bool,
    default = True,
    help = "Logs to WandB",
    required = False
)

args = parser.parse_args()

BATCH_SIZE = args.batch_size
CHECKPOINT_WRITE = args.checkpoint_write
DATASET = args.dataset
DEVICE = args.device 
EPOCHS = args.epochs
EPOCH_WRITE = args.epoch_write
LEARNING_RATE = args.learning_rate
MODEL = args.model
VERBOSE = args.verbose
WANDB = args.wandb

FOLDER_NAME = f"{DATASET}_{args.name}"
RUN_NAME = f"{MODEL}_{DATASET}_{args.name}"

##########
# Config #
##########

import torch
import timm.scheduler as sched

from soft_dice_loss import SoftDiceLossV1

################
# Train Config #
################

def bce_kld_loss(output, target):
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    :param args:
    :param kwargs:
    :return:
    """
    recons = output[0]
    # input = args[1]
    mu = output[1]
    log_var = output[2]

    # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
    recons_loss = torch.nn.functional.binary_cross_entropy(recons, target)


    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    # loss = recons_loss + kld_weight * kld_loss
    loss = recons_loss + kld_loss
    # return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
    return loss

TRAIN_CONFIG = {
  # "criterion": torch.nn.BCEWithLogitsLoss,
  "criterion": torch.nn.BCELoss,
    # "criterion": lambda: bce_kld_loss,
  # "criterion": SoftDiceLossV1,
  "num_workers": 0,
  "optimizer": torch.optim.Adam,
  "scheduler": sched.CosineLRScheduler, 
  "train_fraction": 0.75,
  "test_fraction": 0.25,
  "weight_decay": 0.0,
}

train_config = {
  **TRAIN_CONFIG,
  "batch_size_train": BATCH_SIZE,
  "batch_size_test": BATCH_SIZE,
  "checkpoint_write": CHECKPOINT_WRITE,
  "device": DEVICE,
  "epochs": EPOCHS,
  "epoch_write": EPOCH_WRITE,
  "learning_rate": LEARNING_RATE,
}

dataset_config = {
  **load_config("./dataset.cfg"),
  "dataset": DATASET 
}

##################
# Results Folder #
##################

results_folder = f"./results/{MODEL}/{FOLDER_NAME}"
if not os.path.isdir("./results"):
    os.mkdir("./results")
if not os.path.isdir(f"./results/{MODEL}"):
    os.mkdir(f"./results/{MODEL}")
if not os.path.isdir(results_folder):
    os.mkdir(results_folder)

write_config(args.__dict__, f"{results_folder}/run.cfg", verbose = VERBOSE)
write_config(dataset_config, f"{results_folder}/dataset.cfg", verbose = VERBOSE)

if VERBOSE:
    pprint(load_config(f"{results_folder}/run.cfg"))
    pprint(load_config(f"{results_folder}/dataset.cfg"))

########
# Copy #
########

# Copy Initialization Files
shutil.copyfile(f"./initialize.py", f"./{results_folder}/initialize.py")

for file in os.listdir(f"./models/{MODEL}"):
    # Avoids __pycache__ folder
    if not os.path.isdir(f"./models/{MODEL}/{file}"):
        shutil.copyfile(f"./models/{MODEL}/{file}", f"./{results_folder}/{file}")

for file in os.listdir(f"./metrics"):
    shutil.copyfile(f"./metrics/{file}", f"./{results_folder}/{file}")

shutil.copyfile(f"./dataset.py", f"./{results_folder}/dataset.py")

######################
# Test / Train Split #
######################

train_indexes, test_indexes = split_dataset(
    train_config["train_fraction"],
    train_config["test_fraction"]
)

np.save(f"{results_folder}/train_indexes.npy", train_indexes)
np.save(f"{results_folder}/test_indexes.npy", test_indexes)

########################
# Train / Test Dataset #
########################

train_dataset = Dataset(dataset_config, indexes = train_indexes)
test_dataset = Dataset(dataset_config, indexes = test_indexes)

if VERBOSE:
    print(f"train length: {len(train_dataset)}, test length: {len(test_dataset)}")

####################
# Initialize Model #
####################

model = initialize_model(MODEL, results_folder, verbose = VERBOSE)

model.to(DEVICE)

channels = dataset_config["channels"]
frame_length = dataset_config["frame_length"]

model_summary = summary(
    model,
    input_size=(BATCH_SIZE, channels, frame_length, 64, 64),
)

with open(f"{results_folder}/model_summary.txt", "w") as f:
    f.write(str(model_summary))

if VERBOSE:
    print(str(model_summary))

#########
# WandB #
#########

if WANDB:
    config = {
        "dataset": dataset_config,
        "model": load_config(f"{results_folder}/model.cfg"),
        "run": load_config(f"{results_folder}/run.cfg"),
        "train": train_config,
    }
    project = "pyrometry-porosity"
    wandb.init(name = RUN_NAME, project = project, config = config)
    wandb.define_metric("train_loss", summary="min")
    wandb.define_metric("test_loss", summary="min")
    wandb.define_metric("learing_rate", summary="min")
    wandb.watch(model)

###############
# Train Model #
###############
    
train(
    config = train_config,
    model = model,
    model_name = MODEL,
    train_dataset = train_dataset,
    test_dataset = test_dataset,
    results_folder = results_folder,
    use_wandb = WANDB,
)

if WANDB: wandb.finish()
