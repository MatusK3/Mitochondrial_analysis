import pandas as pd
import matplotlib.pyplot as plt



def plot_csv_data(x_axis, val_acc):
    # 1. Load the data
    # df = pd.read_csv(file_path)
    
    # 2. Create the plot
    plt.figure(figsize=(10, 6))
    
    # plt.plot(x_axis, sparsity, label="sparsity", marker='o')
    for label, value in val_acc.items():
        plt.plot(x_axis, value, label=label, marker='o')

    # for col in y_cols:
    #     if col == 'sparsity':
    #         df[col] /= 100
    #     plt.plot(df[x_col], df[col], label=col, marker='o')
    
    # 3. Add formatting
    # plt.xlabel("x_axis")
    plt.xlabel("epoch")

    plt.ylabel('val_f1_macro')
    plt.ylim((0,1))

    plt.title("Training validation")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Show the result
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # labels = ["9", "18", "34"]
    labels = ["18"]

    prune_step = 0
    x_axis = None
    val_acc = dict()
    for i in labels: 
        csv_file = f'results/cnn/resnet{i}/p{prune_step}_resnet{i}_training_log.csv'
        df = pd.read_csv(csv_file)
        if x_axis is None:
            x_axis = df['epoch']
        val_acc[f"resnet{i}"] = df['val_f1_macro']
    
    plot_csv_data(x_axis, val_acc)
