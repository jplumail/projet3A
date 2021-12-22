# This file should be run after predict.py
# It computes metrics and saves them in metrics.txt

import os
from tqdm import tqdm
import rasterio as rio
import numpy as np

from floatingobjects.data import FloatingSeaObjectDataset


def get_predictions(path):
    with rio.open(path) as src:
        image = src.read()
    return image[0].astype(float) / 255

def get_labels(dataset):
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
        tn = np.trace(confusion_matrix) - tp
        out[c, 0] = tp / (fp + tp) # Precision
        out[c, 1] = tp / (fn + tp) # Recall
        out[c, 2] = 2 / (1/out[c,0] + 1/out[c,1]) # F1-score
    return out



if __name__ == "__main__":

    root_data = "floatingobjects\\data"
    predictions_path = "predictions"

    for model in ["unet", "manet"]:
        model_path = os.path.join(predictions_path, model)

        for fold in [1, 2]:
            fold_path = os.path.join(model_path, str(fold))
            ds = FloatingSeaObjectDataset(root_data, fold="test", foldn=fold, hard_negative_mining=False, use_l2a_probability=0)

            for classifier in ["no-classifier", "classifier"]:
                classifier_path = os.path.join(fold_path, classifier)
                os.makedirs(classifier_path, exist_ok=True)

                classifier_model_path = f"models\\checkpoint-fold{fold}.pt" if classifier=="classifier" else None
                cm = np.zeros((2,2), dtype=int)

                # Iterate over regions
                for dataset in tqdm(ds.datasets):
                    # Get predictions
                    region_path = os.path.join(classifier_path, f"{dataset.region}.tif")
                    y_pred = get_predictions(region_path)

                    # Get labels
                    labels = get_labels(dataset)

                    t = 0.3
                    y_pred = y_pred.ravel() > t
                    labels = labels.ravel()

                    # Compute the confusion matrix
                    cm += np.bincount(labels * 2 + y_pred, minlength=4).reshape(2, 2).astype(int)
                
                with open(os.path.join(predictions_path, "metrics.txt"), "a") as f:
                    f.write(f"Model {model}, fold n°{fold}, {classifier}\n")
                    f.write("Confusion matrix :\n")

                    # Output confusion matrix
                    f.write(repr(cm)+"\n")

                    # Output Precision/recall/F1-score for water and floating objects
                    metrics = get_metrics(cm)
                    for i, c in enumerate(["water", "floating objects"]):
                        f.write(f"\nprecision {c} : {metrics[i,0]:.3f}\n")
                        f.write(f"recall {c} : {metrics[i,1]:.3f}\n")
                        f.write(f"F1 {c} : {metrics[i,2]:.3f}\n")
                    f.write("\n----------\n")
                    

