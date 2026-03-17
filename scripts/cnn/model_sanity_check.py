from fetch_data import fetch_mini
import torch
import torch.nn as nn

from experiment_loader import ExperimentLoader
from experiments.experiments_config import experiments


def run_sanity_check(exp: ExperimentLoader):
    model = exp.model
    loss_fn = exp.loss_fn
    # optimizer = exp.optimizer
    # scheduler=exp.scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler=None

    device = exp.device

    dataloader = fetch_mini()
    # sanity check
    for e in range(500):
        model.train()
        for i, batch in enumerate(dataloader):
            x, y = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, batch in enumerate(dataloader):
                x, y = batch[0].to(device), batch[1].to(device)

                out = model(x)
                loss = loss_fn(out, y)
                acc = torch.sum(torch.argmax(out, dim=-1) == y)
                correct += acc.item()
                total += len(batch[1])

        acc = correct / total
        print(f"At step {e}: loss: {loss.item()}; acc: {acc}")
        if(acc == 1):
            print("sanity checked")
            break

        if scheduler is not None:
            scheduler.step()
    else:
        print("not so great")


if __name__ == "__main__":
    for exp in experiments:
        exp = ExperimentLoader(exp)
        print(f"\n\n\n{exp.name}")
        run_sanity_check(exp)
    






