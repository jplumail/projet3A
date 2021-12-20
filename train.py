import os

import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as T
from tqdm import tqdm

from floatingobjects.model import get_model


class ShipDataset(data.Dataset):
    def __init__(self, path, foldn=1, device="cpu"):
        super().__init__()
        ships = np.load(os.path.join(path, "ships.npz"))
        floatingobjects = np.load(os.path.join(path,f"floatingobjects-fold{foldn}.npz"))
        array_ships = (ships["data"].astype(float) - ships["center"][:,None,None]) / ships["scale"][:,None,None]
        
        self.floatingobjects_center = floatingobjects["center"]
        self.floatingobjects_scale = floatingobjects["scale"]
        array_floatingobjects = (floatingobjects["data"].astype(float) - self.floatingobjects_center[:,None,None]) / self.floatingobjects_scale[:,None,None]
        self.X = np.concatenate([array_ships, array_floatingobjects], axis=0).astype(np.float32)
        self.y = np.concatenate([ships["labels"], floatingobjects["labels"]], axis=0).astype(np.float32)
        self.X = torch.from_numpy(self.X).to(device)
        self.y = torch.from_numpy(self.y).to(device)
        self.transforms = T.RandomHorizontalFlip()
    
    def __getitem__(self, idx):
        x = self.transforms(self.X[idx])
        return x, self.y[idx]
    
    def __len__(self):
        return self.X.shape[0]


def create_loaders(path, foldn, device):
    ds = ShipDataset(path, foldn, device)
    train_len = len(ds)-int(0.4*len(ds))
    valid_len = int(0.4*len(ds)) - int(0.2*len(ds))
    test_len = int(0.2*len(ds))
    train_ds, valid_ds, test_ds = data.random_split(ds, [train_len, valid_len, test_len])

    # Define weights for sampling
    w_ship = len(ds) / ds.y.sum()
    w = [w_ship if y else 1. for _,y in train_ds]
    sampler = data.WeightedRandomSampler(w, num_samples=1000)

    # data loader
    train_loader = data.DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
    valid_loader = data.DataLoader(valid_ds, batch_size=256, num_workers=0)
    return train_loader, valid_loader


def do_epoch(i, model, train_loader, valid_loader, criterion, optimizer, device):
    # Train
    model.train()
    for x_train, y_train in train_loader:
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Evaluation
    model.eval()
    y_evals, y_preds = [], []
    for x_eval, y_eval in valid_loader:
        with torch.no_grad():
            x_eval = x_eval.to(device)
            y_pred = model(x_eval)
        y_pred = y_pred.cpu()
        y_evals.append(y_eval)
        y_preds.append(y_pred)
    y_evals = torch.cat(y_evals, axis=0).cpu().numpy().flatten().astype(bool)
    y_preds = torch.cat(y_preds, axis=0).cpu().numpy().flatten()
    y_preds = y_preds > 0.5
    return classification_report(y_evals, y_preds, target_names=["floating objects","ships"], output_dict=True)


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using", device)
    for foldn in [1, 2]:
        print("Training on fold", foldn)
        # Data
        train_loader, valid_loader = create_loaders("dataset_classifier", foldn, device)
        ds = train_loader.dataset.dataset
        center = torch.from_numpy(ds.floatingobjects_center)
        scale = torch.from_numpy(ds.floatingobjects_scale)

        # Classifier
        clf = get_model("classifier", pretrained=False)
        clf = clf.to(device)

        # Criterion
        criterion = nn.BCELoss()

        # Optimizer
        optimizer = torch.optim.Adam(clf.parameters())

        epochs = 10
        best_f1 = 0
        t = tqdm(range(epochs), desc=f"Best F1 : {best_f1}, current F1 : 0")
        for i in t:
            report = do_epoch(i, clf, train_loader, valid_loader, criterion, optimizer, device)
            if report["ships"]["f1-score"] > best_f1:
                torch.save({
                    "state_dict": clf.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "report_validset": report,
                    "center": center,
                    "scale": scale
                }, f"models/checkpoint-fold{foldn}.pt")
                best_f1 = report["ships"]["f1-score"]
            t.set_description(f"Best F1 : {best_f1}, current F1 : {report['ships']['f1-score']}", refresh=True)