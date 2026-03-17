
import torch
import torch.nn.utils.prune as prune
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import copy

from sklearn.metrics import f1_score

from experiment_loader import ExperimentLoader
from experiments.experiments_config import experiments

def is_model_pruned(model):
    for module in model.modules():
        if prune.is_pruned(module):
            return True
    return False

def save_model_flexible(model, path):
    if is_model_pruned(model):
        model_to_save = copy.deepcopy(model)
        for name, module in model_to_save.named_modules():
            if prune.is_pruned(module):
                # This 'bakes' the mask into the weights
                if hasattr(module, 'weight_mask'):
                    prune.remove(module, 'weight')
                # Optional: If you also prune biases (rare in ResNet, but good for safety)
                if hasattr(module, 'bias_mask'):
                    prune.remove(module, 'bias')
        torch.save(model_to_save.state_dict(), path)

    else:
        torch.save(model.state_dict(), path)






def run_training(exp: ExperimentLoader, save_model=True, prune_iteration=0):
    model = exp.model
    loss_fn = exp.loss_fn
    
    optimizer, scheduler = exp.get_optimizer_and_scheduler()
    device = exp.device
    dataloader_train = exp.get_train_loader()
    dataloader_val =  exp.get_val_loader()
    epochs = exp.epochs

    best_epoch = 0
    best_val = 0
    # best_val_loss = np.inf
    # best_val_acc = 0
    best_model_file = exp.get_model_path(prune_iteration)
    log_csv_file = exp.get_training_log_path(prune_iteration)

    epoch_train_losses = []
    epoch_val = []
    # epoch_val_losses = []
    # epoch_val_accs = []

    pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    for e in pbar:
        train_losses = []
        # val_losses = []

        model.train()

        train_pbar = tqdm(dataloader_train, desc=f"Epoch {e}/{epochs-1} [Train]", leave=False)
        for batch in train_pbar:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
            
            train_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        model.eval()
        with torch.no_grad():
            # correct = 0
            # total = 0
            pred_y = []
            true_y = []
            for batch in dataloader_val:
                x, y = batch[0].to(device), batch[1].to(device)

                out = model(x)
                pred = torch.argmax(out, dim=-1)

                pred_y.extend(pred.cpu().numpy())
                true_y.extend(y.cpu().numpy())
        
        val_f1_macro = f1_score(true_y, pred_y, average='macro')
                
        #         loss = loss_fn(out, y)

        #         acc = torch.sum(torch.argmax(out, dim=-1) == y)
        #         correct += acc.item()
        #         total += y.size(0)
        #         val_losses.append(loss.item())

        # val_acc = correct / total

        epoch_val.append(val_f1_macro)
        epoch_train_losses.append(np.mean(train_losses))
        # epoch_val_losses.append(np.mean(val_losses))
        # epoch_val_accs.append(val_acc)

        if scheduler is not None:
            scheduler.step()

        pbar.set_postfix({
            "tr_loss": f"{epoch_train_losses[-1]:.3f}",
            "val": f"{epoch_val[-1]:.3f}",
            # "val_loss": f"{epoch_val_losses[-1]:.3f}",
            # "val_acc": f"{val_acc:.3f}",
            "top_e:": best_epoch,
            "best_val": best_val
            # "top_loss": f"{best_val_loss:.3f}",
            # "top_acc": f"{best_val_acc:.3f}"
        })

        # if epoch_val_losses[-1] < best_val_loss:
        #     best_val_loss = epoch_val_losses[-1]
            # best_val_acc = val_acc
        if epoch_val[-1] > best_val:
            best_val = epoch_val[-1]
            best_epoch = e
            if save_model:
                save_model_flexible(model, best_model_file)
                # torch.save(model.state_dict(), best_model_file)

        # Early stoping if Nan
        # nan_in_grads = any(p.grad is not None and torch.isnan(p.grad).any() for p in model.parameters())
        # if np.isnan(epoch_train_losses[-1]) or np.isnan(epoch_val[-1]) or nan_in_grads:
        #     print("Nan, occured in loss or gradient => Stopping training")
        #     break
    
    if save_model:
        log_df = pd.DataFrame({
            "epoch": np.arange(len(epoch_train_losses)),
            "train_loss": epoch_train_losses,
            "val_f1_macro": epoch_val,
            # "val_loss": epoch_val_losses,
            # "val_acc": epoch_val_accs,
            "best_model_epoch": np.arange(epochs) == best_epoch
        })
        log_df.to_csv(log_csv_file, index=False)
        print(f"\n Training finished. \nLog saved to {log_csv_file}")
        print(f"Best model saved to {best_model_file}")



if __name__ == "__main__":
    save_model=True
    # save_model=False

    for exp in experiments:
        exp = ExperimentLoader(exp)
        print(f"\n\n\n{exp.name}")
        run_training(exp)



