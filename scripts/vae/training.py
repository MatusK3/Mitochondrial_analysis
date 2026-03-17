# ml_vision env:
# conda activate .snakemake/conda/7f69f82cf5e2d5fc9f5eb4294544276a_



import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiment_loader import ExperimentLoader
from experiments_config import experiments

def get_beta(current_epoch, warmup_epochs, target_beta=0.1):
    """Linearly increases beta from 0 to target_beta over warmup_epochs."""
    if current_epoch >= warmup_epochs:
        return target_beta
    
    # Calculate the fraction of the warmup phase completed
    step = current_epoch / warmup_epochs
    return target_beta * step


def run_training(exp: ExperimentLoader, save_model=True):
    model = exp.model
    loss_fn = exp.loss_fn
    
    optimizer, scheduler = exp.get_optimizer_and_scheduler()
    device = exp.device
    dataloader_train = exp.get_train_loader()
    # dataloader_val =  exp.get_val_loader()
    epochs = exp.epochs

    model_file = exp.get_model_path()
    log_csv_file = exp.get_training_log_path()



    target_beta = 1
    warmup_epochs = int(epochs * 0.4)   



    epoch_train_losses = [[], [], []]

    pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    for e in pbar:
        train_losses = [[], [], []]

        model.train()

        current_beta = get_beta(e, warmup_epochs, target_beta)

        train_pbar = tqdm(dataloader_train, desc=f"Epoch {e}/{epochs-1} [Train]", unit="batch", leave=False)
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


    if save_model:
        log_df = pd.DataFrame({
            "epoch": np.arange(len(epoch_train_losses[0])),
            "train_loss": epoch_train_losses[0],
            "recon_loss": epoch_train_losses[1], 
            "kld_loss": epoch_train_losses[2]
        })
        
        torch.save(model.state_dict(), model_file)
        log_df.to_csv(log_csv_file, index=False)
        print(f"model saved to {model_file}")
        print(f"\n Training finished. \nLog saved to {log_csv_file}")
        


if __name__ == "__main__":
    save_model=True
    # save_model=False

    for exp in experiments:
        exp = ExperimentLoader(exp)
        print(f"\n\n\n{exp.name}")
        run_training(exp)



