import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd

from experiment_loader import ExperimentLoader
from fetch_data import fetch_test_dataloader
from experiments.experiments_config import experiments

import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def evaluate_model(exp: ExperimentLoader, dataloader, plot_results=False):
    # get class names
    DATSET_PATH = "data/mito"
    config_path = f"{DATSET_PATH}/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    class_names = list(config["class_to_int_mapping"].keys())

    # evaluation: 
    name = exp.name
    device = exp.device
    # loss_fn = exp.loss_fn
    model = exp.model

    model.eval()
    
    # total_loss = 0.0
    # correct = 0
    # total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, y = batch[0].to(device), batch[1].to(device)

            out = model(x)
            # loss = loss_fn(out, y)
            # total_loss += loss.item() * x.size(0)

            predicted = torch.argmax(out, dim=-1)
            # acc = torch.sum(predicted == y)
            # correct += acc.item()
            # total += len(batch[1])

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # avg_loss = total_loss / total
    # acc = correct / total

    # print(f"{name}")
    # print(f"    Test Los: {avg_loss}")
    # print(f"    Test Acc: {acc}")
    # return avg_loss, acc  

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    print(class_names)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    # weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    # return weighted_f1

    if plot_results:
        weighted_f1 = report['macro avg']['f1-score']

        # report_df = pd.DataFrame(report).transpose().round(3)
        # report_df.loc['accuracy', ['precision', 'recall']] = None
        # print(f"{report_df}\n{exp.name}; Weighted F1: {weighted_f1:.4f}")
        print(f"\n\n{exp.name}; macro F1: {weighted_f1:.4f}")
        print(classification_report(y_true, y_pred, target_names=class_names))
        

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'{exp.name} Confusion Matrix')
        plt.show()
    else:
        return report
    



if __name__ == "__main__":
    # results = []
    dataloader_test = fetch_test_dataloader()
    for exp in experiments:
        exp = ExperimentLoader(exp)
        exp.load_model()
        evaluate_model(exp, dataloader_test, plot_results=True)
        # avg_loss, acc = evaluate_model(exp, dataloader_test)
        # results.append((exp.name, avg_loss, acc))
    

    # n=5
    # top_n = sorted(results, key=lambda x: x[2], reverse=True)[:n] # BY ACCURACY
    # for key, avg_loss, acc in top_n:
    #     print(key, acc)
    

# resnet9; Weighted F1: 0.6624
# resnet18; Weighted F1: 0.6776
# resnet34; Weighted F1: 0.5785