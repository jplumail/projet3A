# This file should be run after predict.py
# It computes metrics and saves them in metrics.txt

import os
from tqdm import tqdm
import rasterio as rio
import numpy as np
import torch
from torch.utils.data import Subset
import argparse

from floatingobjects.data import FloatingSeaObjectDataset
from floatingobjects.model import get_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="/data")
    parser.add_argument('--snapshot-path', type=str)
    parser.add_argument('--metrics-path', type=str)
    parser.add_argument('--test-negative-patches', action="store_true")
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--model', type=str, default="unet")
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--device', type=str, choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    return args

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
                  

def main(args):
    threshold = 0.03
    device = args.device
    
    for fold in [1, 2]:

        dataset = FloatingSeaObjectDataset(
            args.data_path, fold="test", foldn=fold, transform=None, output_size=args.image_size, use_l2a_probability=0,
            hard_negative_mining=False
        )

        if args.test_negative_patches:
            indices = dataset.filter_hnm() # indices of negatives patches only
            dataset = Subset(dataset, indices)
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=args.num_workers)

        m = get_model(args.model, pretrained=False)
        m.load_state_dict(torch.load(args.snapshot_path)["model_state_dict"])
        m = m.to(device)
        m.eval()

        for classifier in ["no-classifier", "classifier"]:

            classifier_model_path = f"models\\checkpoint-fold{fold}.pt" if classifier=="classifier" else None
            # Load classifier
            if classifier_model_path:
                classifier_model = get_model("classifier")
                checkpoint = torch.load(classifier_model_path)
                classifier_model.load_state_dict(checkpoint["state_dict"])
                classifier_model = classifier_model.to(device)
                classifier_model.eval()
                center = checkpoint["center"].to(device)
                scale = checkpoint["scale"].to(device)
            else:
                classifier_model = None

            cm = np.zeros((2,2), dtype=int)
            for idx, (image, target, _) in tqdm(enumerate(loader), total=len(loader)):
                image = image.to(device, dtype=torch.float)
                image *= 1e-4
                with torch.no_grad():
                    # forward pass: compute predicted outputs by passing inputs to the model
                    logits = m.forward(image)
                    output = torch.sigmoid(logits)[:, 0]

                    if classifier == "classifier":
                        # Image transformation
                        image *= 1e4
                        image -= center[:,None,None]
                        image /= scale[:,None,None]

                        # Apply sliding windows
                        classifier_scores = classifier_model.sliding_windows(image, stride=8)
                        mask_ship = classifier_scores > 0.99
                        output[mask_ship] = 0.

                    y = (output > threshold).cpu().view(-1).numpy()
                    y_true = target.cpu().view(-1).numpy().astype(bool)
                    cm += np.bincount(y_true * 2 + y, minlength=4).reshape(2, 2).astype(int)
            
            yield model, fold, classifier, cm



if __name__ == "__main__":
    args = parse_args()

    metrics_path = args.metrics_path
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    for model, fold, classifier, cm in main(args):
        with open(os.path.join(metrics_path, "predictions-nohnm3.txt"), "a") as f:
            f.write(f"Model {model}, fold n??{fold}, {classifier}\n")
            f.write("Confusion matrix :\n")

            # Take only some pixels, this is how ESA did their tests
            N = 20000
            cm = cm.astype(float)
            n_true = cm.sum(axis=1)
            cm = (N * cm / n_true[:,None]).astype(int)

            # Output confusion matrix
            f.write(repr(cm)+"\n")

            f.write(f"\nfalse positive rate : {cm[0,1] / (cm[0,0]+cm[0,1]):.3f}")

            # Output Precision/recall/F1-score for water and floating objects
            metrics = get_metrics(cm)
            for i, c in enumerate(["water", "floating objects"]):
                f.write(f"\nprecision {c} : {metrics[i,0]:.3f}\n")
                f.write(f"recall {c} : {metrics[i,1]:.3f}\n")
                f.write(f"F1 {c} : {metrics[i,2]:.3f}\n")
            f.write("\n----------\n")


