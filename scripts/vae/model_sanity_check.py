from fetch_data import fetch_mini
import numpy as np
from tqdm.auto import tqdm
from experiment_loader import ExperimentLoader
from experiments_config import experiments
from models import VAE, vae_loss
import torch


def get_beta(current_epoch, warmup_epochs, target_beta=0.1):
    """Linearly increases beta from 0 to target_beta over warmup_epochs."""
    if current_epoch >= warmup_epochs:
        return target_beta
    
    # Calculate the fraction of the warmup phase completed
    step = current_epoch / warmup_epochs
    return target_beta * step


def run_sanity_check():
    # optimizer, scheduler = exp.get_optimizer_and_scheduler()
    # device = exp.device
    epochs = 200

    target_beta = 1
    warmup_epochs = int(epochs * 0.4)   


    device = "cuda"
    model = VAE(latent_dim=268, blocks_per_layer=1)
    model.to(device)
    loss_fn = vae_loss
    optimizer = torch.optim.Adam(model.parameters() , lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # scheduler = None

    
    sample_size = 8
    dataloader= fetch_mini(sample_size)
    
    epoch_train_losses = [[], [], []]
    
    # from models import PerceptualLoss
    # percploss_model = PerceptualLoss()
    # percploss_model.to(device)

    pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    for e in pbar:
        train_losses = [[], [], []]

        model.train()

        current_beta = get_beta(e, warmup_epochs, target_beta)

        train_pbar = tqdm(dataloader, desc=f"Epoch {e}/{epochs-1} [Train]", leave=False)
        for batch in train_pbar:
            x = batch[0].to(device)

            optimizer.zero_grad()

            recon_batch, mu, logvar = model(x)

            loss, recon_loss, kld_loss = loss_fn(recon_batch, x, mu, logvar, current_beta)
            loss.backward()

            train_losses[0].append(loss.item())
            train_losses[1].append(recon_loss.item())
            train_losses[2].append(kld_loss.item())
            optimizer.step()
            
            train_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "recon_loss": f"{recon_loss.item():.4f}", "kld_loss": f"{kld_loss.item():.4f}"})

        # epoch_train_losses.append(np.mean(train_losses))
        epoch_train_losses[0].append(np.mean(train_losses[0]))
        epoch_train_losses[1].append(np.mean(train_losses[1]))
        epoch_train_losses[2].append(np.mean(train_losses[2]))

        if scheduler is not None:
            scheduler.step()

        pbar.set_postfix({
            "tr_loss": f"{epoch_train_losses[0][-1]:.4f}",
            "recon_loss": f"{epoch_train_losses[1][-1]:.4f}", 
            "kld_loss": f"{epoch_train_losses[2][-1]:.4f}"
        })
    
    import matplotlib.pyplot as plt 
    import torch.nn.functional as F

    with torch.no_grad():
        dataloader= fetch_mini(sample_size)
        for batch in dataloader:
            x, y = batch[0].to(device), batch[1]
            recon, _, _ = model(x)

        rows = min([3, sample_size])
        print(rows)
        fig_size = 4
        fig, axes = plt.subplots(rows, 2, figsize=(fig_size*2, fig_size*rows))
        if rows == 1: axes = [axes]
        for i in range(rows):
            input = np.moveaxis(x.cpu().numpy()[i], 0, -1)
            output = np.moveaxis(F.sigmoid(recon).cpu().numpy()[i], 0, -1)

            axes[i][0].imshow(input)
            axes[i][0].set_title(f"{y}, input")
            # axes[0].axis("off")

            axes[i][1].imshow(output)
            axes[i][1].set_title(f"{y}, output")
            # axes[1].axis("off")

        plt.tight_layout()
        plt.show()




if __name__ == "__main__":

    # exp = ExperimentLoader(experiments[0])
    # model = exp.model
    # import torch
    # dummy_input = torch.randn(1, 1, 128, 128).to(exp.device) # Batch, Channel, Height, Width
    # recon, mu, logvar = model(dummy_input)

    # print(f"Input Shape: {dummy_input.shape}")
    # print(f"Output Shape: {recon.shape}") # Should be [1, 1, 64, 64]
    # print(f"Latent mu Shape: {mu.shape}")   # Should be [1, 128]

    # loss = exp.loss_fn(recon, dummy_input, mu, logvar)
    # print(f"loss: {loss}")

    run_sanity_check()

    






