# This file should be ran after predict.py
# It computes metrics and saves them in metrics.txt

import os
from floatingobjects.transforms import get_transform
from tqdm import tqdm
import rasterio as rio
import numpy as np
import torch

from floatingobjects.data import FloatingSeaObjectDataset
from floatingobjects.model import get_model


def get_predictions(path):
    with rio.open(path) as src:
        image = src.read()
    return image[0].astype(float) / 255

def get_labels(dataset, y_pred):
    left, bottom, right, top = dataset.imagebounds
    window = rio.windows.from_bounds(left, bottom, right, top, dataset.imagemeta["transform"])
    with rio.open(dataset.imagefile) as src:
        win_transform = src.window_transform(window)
    labels = rio.features.rasterize(dataset.rasterize_geometries, all_touched=True,
                    transform=win_transform, out_shape=y_pred.shape)
    labels = labels.astype(bool)
    return labels

def get_metrics(confusion_matrix):
    n_classes = len(confusion_matrix)
    out = np.zeros((n_classes, 3)) # store precision+recall+f1-score
    for c in range(n_classes):
        tp = confusion_matrix[c, c]
        fp = confusion_matrix[:, c].sum() - tp
        fn = confusion_matrix[c, :].sum() - tp
        out[c, 0] = tp / (fp + tp) # Precision
        out[c, 1] = tp / (fn + tp) # Recall
        out[c, 2] = 2 / (1/out[c,0] + 1/out[c,1]) # F1-score
    return out


def predictions_all(root_data, predictions_path):

    for model in ["unet", "manet"]:
        model_path = os.path.join(predictions_path, model)

        for fold in [1, 2]:
            fold_path = os.path.join(model_path, str(fold))
            ds = FloatingSeaObjectDataset(root_data, fold="test", foldn=fold, hard_negative_mining=False, use_l2a_probability=0)

            for classifier in ["no-classifier", "classifier"]:
                classifier_path = os.path.join(fold_path, classifier)
                os.makedirs(classifier_path, exist_ok=True)

                cm = np.zeros((2,2), dtype=int)
                # Iterate over regions
                for dataset in tqdm(ds.datasets):
                    # Get predictions
                    region_path = os.path.join(classifier_path, f"{dataset.region}.tif")
                    y_pred = get_predictions(region_path)

                    # Get labels
                    labels = get_labels(dataset, y_pred)

                    print(y_pred.shape)

                    t = 0.3
                    y_pred = y_pred.ravel() > t
                    labels = labels.ravel()

                    # Compute the confusion matrix
                    cm += np.bincount(labels * 2 + y_pred, minlength=4).reshape(2, 2).astype(int)
                
                yield model, fold, classifier, cm
                
                
                

def predictions_esa(root_data, predictions_path):
    threshold = 0.03
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = get_transform("test")
    for model in ["unet", "manet"]:
        for fold in [1, 2]:
            dataset = FloatingSeaObjectDataset(root_data, fold="test", foldn=fold, transform=None, output_size=128, use_l2a_probability=0.5)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

            m = get_model(model, pretrained=False)
            m.load_state_dict(torch.load(f"models\\{model}-posweight1-lr001-bs160-ep50-aug1-seed{fold-1}.pth.tar")["model_state_dict"])
            m = m.to(device)

            for classifier in ["no-classifier"]:

                classifier_model_path = f"models\\checkpoint-fold{fold}.pt" if classifier=="classifier" else None
                cm = np.zeros((2,2), dtype=int)
                for idx, (image, y_true, _) in tqdm(enumerate(loader), total=len(loader)):
                    image = image.to(device, dtype=torch.float)
                    image *= 1e-4
                    with torch.no_grad():
                        # forward pass: compute predicted outputs by passing inputs to the model
                        logits = m.forward(image)
                        output = torch.sigmoid(logits)

                        # Sans classifier pour l'instant
                        """
                        if self.classifier:
                            mask_floating_objects = y_logits > 0.05
                            tensor = torch.from_numpy(image.astype(np.float32)).float().to(self.device).unsqueeze(0)
                            # Image transformation
                            tensor -= self.center[:,None,None]
                            tensor /= self.scale[:,None,None]

                            # Apply sliding windows
                            classifier_scores = self.classifier.sliding_windows(tensor, mask_floating_objects)
                            mask_ship = classifier_scores > 5e-4
                            y_logits[mask_ship] = 0.
                        """
                        y = (output > threshold).cpu()
                        cm += np.bincount(y_true.ravel() * 2 + y.ravel(), minlength=4).reshape(2, 2).astype(int)
                
                yield model, fold, classifier, cm



if __name__ == "__main__":

    root_data = "floatingobjects\\data"
    predictions_path = "predictions"


    for model, fold, classifier, confusion_matrix in predictions_esa(root_data, predictions_path):
        with open(os.path.join("predictions-esa", "metrics.txt"), "a") as f:
            f.write(f"Model {model}, fold n°{fold}, {classifier}\n")
            f.write("Confusion matrix :\n")

            # Output confusion matrix
            f.write(repr(confusion_matrix)+"\n")

            # Output Precision/recall/F1-score for water and floating objects
            metrics = get_metrics(confusion_matrix)
            for i, c in enumerate(["water", "floating objects"]):
                f.write(f"\nprecision {c} : {metrics[i,0]:.3f}\n")
                f.write(f"recall {c} : {metrics[i,1]:.3f}\n")
                f.write(f"F1 {c} : {metrics[i,2]:.3f}\n")
            f.write("\n----------\n")


