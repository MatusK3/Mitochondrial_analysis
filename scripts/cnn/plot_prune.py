import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_csv_data(x_axis, sparsity, f1_scores):
    # 1. Load the data
    # df = pd.read_csv(file_path)
    
    # 2. Create the plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(x_axis, sparsity, label="sparsity", marker='o')
    for label, value in f1_scores.items():
        plt.plot(x_axis, value, label=f"{label} test_loss", marker='o')

    # for col in y_cols:
    #     if col == 'sparsity':
    #         df[col] /= 100
    #     plt.plot(df[x_col], df[col], label=col, marker='o')
    
    # 3. Add formatting
    # plt.xlabel("x_axis")
    plt.xlabel("prune step")

    plt.ylabel('%')
    plt.ylim((0,1))

    plt.title("Prune: tsesting_eval/sparsity")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Show the result
    plt.tight_layout()
    plt.show()


def plot_prune_training(model, epochs, val_acc):
    plt.figure(figsize=(10, 6))

    # "viridis" – perceptually uniform, very safe default
    # "plasma" – more contrast, warmer
    # "inferno" / "magma" – darker, dramatic
    # "coolwarm" – good if there’s a “low → high” meaning
    # "cividis" – color-blind friendly
    cmap = plt.cm.coolwarm 
    colors = cmap(np.linspace(0, 1, len(val_acc)))

    for i, value in enumerate(val_acc):
        plt.plot(epochs, value, label=f"prune step{i}", marker='o', color=colors[i])

    plt.xlabel("epoch")

    plt.ylabel('val_acc')
    plt.ylim((0,1))

    plt.title(f"Training of pruned resnet{model}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Show the result
    plt.tight_layout()
    plt.show()
    



if __name__ == "__main__":
    labels = ["9", "18", "34"]
    # f1_scores = dict()
    # x_axis = None
    # sparsity = None
    # for i in labels: 
    #     csv_file = f'results/cnn/resnet{i}/resnet{i}_test_eval.csv'
    #     df = pd.read_csv(csv_file)
    #     if x_axis is None:
    #         x_axis = df['prune_step']
    #     if sparsity is None:
    #         sparsity = df['sparsity'] / 100
    #     f1_scores[f"resnet{i}"] = df['weighted avg_f1-score']
    
    # plot_csv_data(x_axis, sparsity, f1_scores)


    epochs = None
    for model in labels:
        data_dir = f"results/cnn/resnet{model}"
        val_acc = []
        for prune_ite in range(15):
            data_path = f"{data_dir}/p{prune_ite}_resnet{model}_training_log.csv"

            data = pd.read_csv(data_path)
            if epochs is None:
                epochs = data["epoch"]
            val_acc.append(data["val_acc"])
        plot_prune_training(model, epochs, val_acc)




