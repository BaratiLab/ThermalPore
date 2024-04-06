import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import wandb

from tqdm import tqdm 
from torch.utils.data import DataLoader

import pyvista as pv
pv.start_xvfb()

def train(
    # Configs
    config,
    model,
    model_name,
    train_dataset,
    test_dataset,
    results_folder,

    # Other
    use_wandb = False,
):

    ##########################################
    # Train / Test / Prediction Data Loaders #
    ##########################################
    
    batch_size_train = config["batch_size_train"]
    batch_size_test = config["batch_size_test"]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=config["num_workers"]
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=True,
        num_workers=config["num_workers"]
    )

    # Same as test but unshuffled and batch size is set to 1.
    prediction_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"]
    )

    ###########
    # Metrics #
    ###########

    train_losses = []
    test_losses = []
    learning_rates = []

    ###########
    # Folders #
    ###########

    folders = [
        "checkpoints",
        "predictions",
        "predictions_raw",
        "learning_rates",
        "learning_rates_plot",
        "losses_plot",
        "losses_test",
        "losses_train",
    ]

    for folder in folders:
        if not os.path.isdir(f"{results_folder}/{folder}"):
            os.mkdir(f"{results_folder}/{folder}")

    #############
    # Intialize #
    #############

    device = config["device"]

    optimizer = config["optimizer"](
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay = config["weight_decay"]
    )
    criterion = config["criterion"]()
    scheduler = config["scheduler"](
        optimizer,
        t_initial = config["epochs"],
        decay_rate = 0.1,
        warmup_lr_init = 1E-6,
        warmup_t = 10,
        t_in_epochs = True
    )

    #################
    # Training Loop #
    #################

    for epoch in tqdm(range(config["epochs"])):

        train_running_loss = 0
        test_running_loss = 0

        #########
        # Train #
        #########

        model.train()

        for pore, _ in train_dataloader:

            pore = pore.to(device)
            output = model(pore)

            loss = criterion(output, pore)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()/len(train_dataloader)

        train_losses.append(train_running_loss)

        np.save(f"{results_folder}/losses_train/last.npy", train_losses)

        #############
        # Scheduler #
        #############

        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)

        np.save(f"{results_folder}/learning_rates/last.npy", learning_rates)

        ########
        # Test #
        ########

        model.eval()

        for pore, _ in test_dataloader:

            pore = pore.to(device)
            output = model(pore)

            loss = criterion(output, pore)

            test_running_loss += loss.item()/len(test_dataloader)

        test_losses.append(test_running_loss)

        np.save(f"{results_folder}/losses_test/last.npy", test_losses)

        #########
        # WandB #
        #########

        log_dict = {
            "learing_rate": current_lr, 
            "train_loss": train_running_loss,
            "test_loss": test_running_loss
        }

        if use_wandb:
            wandb.log(log_dict)

        ##############
        # Checkpoint #
        ##############
        # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
            
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        torch.save(state, f"{results_folder}/checkpoints/last.pth")

        ####################
        # Checkpoint Write #
        ####################

        if epoch % config["checkpoint_write"] == 0:
            torch.save(state, f"{results_folder}/checkpoints/{epoch}.pth")

        ###############
        # Epoch Write #
        ###############

        predictions = []
        predictions_raw = []

        # Saves target and video values during first epoch.
        if epoch == 0:
            targets = []
            pores = []

            for video, target in prediction_dataloader:
                targets.append(target)
                pores.append(video)

            with open(f"{results_folder}/targets.p", "wb") as targets_file:
                pickle.dump(targets, targets_file)

            with open(f"{results_folder}/pores.p", "wb") as pores_file:
                pickle.dump(pores, pores_file)
        
        elif epoch % config["epoch_write"] == 0:

            ##############
            # Raw Values #
            ##############

            np.save(f"{results_folder}/losses_train/{epoch}.npy", train_losses)
            np.save(f"{results_folder}/losses_test/{epoch}.npy", test_losses)
            np.save(f"{results_folder}/learning_rates/{epoch}.npy", learning_rates)

            #############################
            #  Plot Train and Test Loss #
            #############################

            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train')
            plt.plot(test_losses, label='Test')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss')
            plt.legend()
            plt.savefig(f"{results_folder}/losses_plot/{epoch}.png")
            plt.close()

            ########################
            #  Plot Learning Rates #
            ########################

            plt.figure(figsize=(10, 5))
            plt.plot(learning_rates, label='Learning Rates')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rates')
            plt.title('Learning Rates')
            plt.legend()
            plt.savefig(f"{results_folder}/learning_rates_plot/{epoch}.png")
            plt.close()

            ###############
            # Predictions #
            ###############

            # Set the model to evaluation mode
            model.eval()

            if not os.path.isdir(f"{results_folder}/renders"):
                os.mkdir(f"{results_folder}/renders")

            # Loop through the test data
            for index, (pore, _) in enumerate(prediction_dataloader):

                # Move data to the device
                pore = pore.to(device)

                # Forward pass to get predictions
                with torch.no_grad():
                    output = model(pore)

                if model_name in ["vae", "unet_vae"] :
                    prediction_raw = output[0].detach().cpu().numpy()
                
                else:
                    # Convert predictions to a numpy array
                    prediction_raw = output.detach().cpu().numpy()

                predictions_raw.append(prediction_raw)

                prediction = prediction_raw.round()
                predictions.append(prediction)

                if not os.path.isdir(f"{results_folder}/renders/{epoch}"):
                    os.mkdir(f"{results_folder}/renders/{epoch}")

                try:
                    # with open(f"predictions/{epoch}.p", "rb") as predictions_file:
                    #     predictions = pickle.load(predictions_file)

                    # predictions = np.array(predictions)

                    # for index in tqdm(range(test_length)):

                    pore = prediction.squeeze()

                    # pore_id = test_indexes[index] + 1

                    pl = pv.Plotter(off_screen=True)

                    # Manually sets bounds for grid
                    for z in [0, pore.shape[0] - 1]:
                        for y in [0, pore.shape[1] - 1]:
                            for x in [0, pore.shape[2] - 1]:
                                # print(z, y, x)
                                pore[z][y][x] = 10

                    zrng = np.arange(0, pore.shape[0] + 1, 1)
                    yrng = np.arange(0, pore.shape[1] + 1, 1)
                    xrng = np.arange(0, pore.shape[2] + 1, 1)

                    grid = pv.RectilinearGrid(xrng, yrng, zrng)
                    grid.cell_data["Values"] = pore.flatten()
                    threshold = grid.threshold(0.5)
                    mesh = pl.add_mesh(threshold, cmap="coolwarm", show_edges=True)

                    pl.camera_position = "iso"
                    pl.show_grid(
                        n_zlabels = pore.shape[0] + 1,
                        n_ylabels = pore.shape[1] + 1,
                        n_xlabels = pore.shape[2] + 1,
                    )
                    pl.remove_scalar_bar()

                    pl.screenshot(f"{results_folder}/renders/{epoch}/{index}.png")
                    pl.close()

                except Exception as e:
                    print(e)

            # with open(f"{results_folder}/predictions_raw/{epoch}.p", "wb") as predictions_raw_file:
            #     pickle.dump(predictions_raw, predictions_raw_file)

            # with open(f"{results_folder}/predictions/{epoch}.p", "wb") as predictions_file:
            #     pickle.dump(predictions, predictions_file)

