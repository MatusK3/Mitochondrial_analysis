import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
# import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiment_loader import ExperimentLoader
from experiments.experiments_config import experiments
from training import run_training
from evaluate_model import evaluate_model
from fetch_data import fetch_test_dataloader


def prune_resnet_globally(model, amount=0.2):
    """
    Globally prunes Conv2d and Linear weights.
    Ignores BatchNorm layers.
    """
    parameters_to_prune = []
    
    for name, module in model.named_modules():
        # Prune Conv2d and the final Linear layer
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    # Global unstructured pruning (prunes the lowest magnitude weights across the whole net)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )


def rewind_resnet_weights(model, original_state_dict):
    with torch.no_grad():
        for name, module in model.named_modules():
            
            # --- Handle Prunable Layers (Conv2d, Linear) ---
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if prune.is_pruned(module):
                    # If pruned, copy initial weight to 'weight_orig'
                    key = f"{name}.weight"
                    module.weight_orig.copy_(original_state_dict[key])
                else:
                    # If for some reason it wasn't pruned, copy normally
                    key = f"{name}.weight"
                    module.weight.copy_(original_state_dict[key])
                
                # Handle Bias (if present, usually ResNet Conv layers have no bias, but Linear does)
                if module.bias is not None:
                    key = f"{name}.bias"
                    module.bias.copy_(original_state_dict[key])
            
            # --- Handle Batch Normalization ---
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # BatchNorm has learnable affine params (weight/bias)
                # AND non-learnable running stats (running_mean/running_var)
                
                # 1. Rewind learnable parameters
                if module.weight is not None:
                    key = f"{name}.weight"
                    module.weight.copy_(original_state_dict[key])
                if module.bias is not None:
                    key = f"{name}.bias"
                    module.bias.copy_(original_state_dict[key])
                
                # 2. Rewind running statistics
                # Note: These are buffers, not parameters, but they are in state_dict
                key_mean = f"{name}.running_mean"
                key_var = f"{name}.running_var"
                # Some implementations track num_batches_tracked as well
                key_batches = f"{name}.num_batches_tracked"
                
                if key_mean in original_state_dict:
                    module.running_mean.copy_(original_state_dict[key_mean])
                if key_var in original_state_dict:
                    module.running_var.copy_(original_state_dict[key_var])
                if key_batches in original_state_dict:
                    module.num_batches_tracked.copy_(original_state_dict[key_batches])



def run_pruning(exp: ExperimentLoader, num_of_iterations = 10, amount=0.2):
    evaluations = []
    dataloader_test = fetch_test_dataloader()

    initial_state_dict = copy.deepcopy(exp.model.state_dict())

    original_size = sum(p.numel() for p in exp.model.parameters())
    pruned_size = original_size
    param_size_tracker = []

    prune_pbar = tqdm(range(num_of_iterations), desc="Pruning", unit="prune_step")
    for prune_step in prune_pbar:
        # track sparsity
        sparsity = 100 * (1 - pruned_size / original_size)
        param_size_tracker.append([pruned_size, sparsity])

        # train
        prune_pbar.set_postfix({"sparsity": sparsity, "state": "training"})
        run_training(exp, save_model=True, prune_iteration=prune_step)

        # evaluate
        prune_pbar.set_postfix({"sparsity": sparsity, "state": "evaluating"})
        evaluations.append(evaluate_model(exp, dataloader_test))

        # prune
        if prune_step < num_of_iterations - 1:
            prune_pbar.set_postfix({"sparsity": sparsity, "state": "pruning"})
            prune_resnet_globally(exp.model, amount)
            rewind_resnet_weights(exp.model, initial_state_dict)
            # recalculate pruned_size
            pruned_size = sum([torch.count_nonzero(buf).item() for name, buf in exp.model.named_buffers() if 'weight_mask' in name])

        

    # save evaluations
    eval_path = exp.evaluations_path
    final_report = []
    for i, eval in enumerate(evaluations):
        row = {
            'prune_step': i,
            'num_of_param': param_size_tracker[i][0],
            'sparsity': param_size_tracker[i][1]
        }
        for key, metrics in eval.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    row[f"{key}_{metric_name}"] = value
            else:
                row[key] = metrics
        final_report.append(row)
    final_report_df = pd.DataFrame(final_report)
    final_report_df.to_csv(eval_path, index=False)


if __name__ == "__main__":
    for exp in experiments[1:]:
        exp = ExperimentLoader(exp)
        print(f"\n\n\n{exp.name}")
        run_pruning(exp, exp.prune_iter)


    
