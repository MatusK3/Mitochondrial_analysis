import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


from config.config import DATASETS
from feature_loader import load_features, split_numpy_data


if __name__ == "__main__":
    classes = [DATASETS.YPD_SD_Acetate_DAY_3_Acetate, DATASETS.YPD_SD_Acetate_DAY_1_Acetate] 


    dataset, labels = load_features(classes)

    featrure_names = np.array(dataset.columns)
    dataset = dataset.to_numpy()
    # print(featrure_names.shape, dataset.shape)

    # encoder = OneHotEncoder(sparse_output=False)
    # one_hot = encoder.fit_transform(labels.to_frame())
    # print(one_hot.shape)

    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    # labels_decoded = encoder.inverse_transform(labels_encoded)
    # print(labels[0], labels_decoded[0], labels[len(labels)-1], labels_decoded[len(labels)-1])




    x_train, y_train, x_val, y_val, x_test, y_test = split_numpy_data(dataset, labels_encoded, val_percent=0.15, test_percent=0)

    dataloader_train = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.long)
        ), 
        batch_size=16, 
        shuffle=True)
    dataloader_val = DataLoader(
        TensorDataset(
            torch.tensor(x_val, dtype=torch.float32), 
            torch.tensor(y_val, dtype=torch.long)
        ), 
        batch_size=16, 
        shuffle=True)



    print(f"liner model shape: {x_train.shape[1]}, {len(classes)}")
    # model = nn.Sequential(
    #         nn.Linear(x_train.shape[1], x_train.shape[1]*2), nn.ReLU(),
    #         nn.Linear(x_train.shape[1]*2, len(classes))
    #     )
    model = nn.Linear(x_train.shape[1], len(classes))
    model_inf = nn.Sequential(model, nn.Softmax(dim=-1))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lambda_param = 1e-2  

    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_accs = []

    model.to(device)
    for e in range(200):
        train_losses = []
        val_losses = []
        model.train()
        for i, batch in enumerate(dataloader_train):
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            out = model(x)

            loss = loss_fn(out, y)
            loss = loss + lambda_param * model.weight.abs().sum()

            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
            # if i % 100 == 0:
            #     print(f"at step {i} training loss {loss.item()}")

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, batch in enumerate(dataloader_val):
                x, y = batch[0].to(device), batch[1].to(device)

                out = model(x)
                loss = loss_fn(out, y)
                acc = torch.sum(torch.argmax(out, dim=-1) == y)
                correct += acc.item()
                total += len(batch[1])
            val_losses.append(loss.item())

        val_acc = correct / total

        epoch_train_losses.append(np.mean(train_losses))
        epoch_val_losses.append(np.mean(val_losses))
        epoch_val_accs.append(val_acc)

        if e % 10 == 0:
            print(f"epoch {e:4d}; train_mean_loss: {epoch_train_losses[-1]:6f}, val_mean_loss: {epoch_val_losses[-1]:6f}; val_mean_acc: {epoch_val_accs[-1]:6f}")



    weights = model.weight.detach().cpu().numpy()
    print(weights.shape)
    threshold = 1e-5
    for cls, ws in zip(classes, weights):
        print(cls, ws.shape)
        selected = np.where(ws > threshold)[0]
        print(f"    Number of selected features: {len(selected)}/{len(ws)}")
        for selected_id in sorted(selected, key=lambda i: ws[i], reverse=True):
            print(f"        {ws[selected_id]:6f}: {featrure_names[selected_id]}")


    
    # .ravel()
    # selected = np.where(np.abs(weights) > 1e-5)[0]

    # print("Number of selected features:", len(selected))
    # print("Selected feature indices:", selected)
    # print("Selected feature names:", featrure_names[selected])





