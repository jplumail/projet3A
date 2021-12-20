# Create the data needed to train the ship classifier.
# If you want to train it yourself,
# you will additional ship data from Alina's repository : https://github.com/alina2204/contrastive_SSL_ship_detection
# Download this archive : https://drive.google.com/file/d/1zDgz6wr5kxikPR7o9nJ2IjMcaqwtiLLu/view
# and make sure that the SHIPS_PATH variable points to the dataset_npy folder

import os

import numpy as np
from scipy.ndimage.measurements import label
from tqdm import tqdm
import rasterio as rio
from sklearn.preprocessing import RobustScaler

from floatingobjects.data import FloatingSeaObjectDataset


l1cbands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
l2abands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

SHIPS_PATH = "S2SHIPS\\dataset_npy"
FLOATINGOBJECTS_PATH = "data"
DATASET_PATH = "dataset_classifier"
os.makedirs(DATASET_PATH, exist_ok=True)

image_size = 16


def get_fname(f):
    l = f.split('.')[0].split('_')
    l[1] = "ships"
    return "_".join(l)


def create_ships():
    list_files = os.listdir(SHIPS_PATH)

    N = 0
    ships = []
    labels = []
    metrics = {
        "number_pixels": [],
        "mean": [],
        "var": []
    }
    pixels = []
    for f in list_files:
        ds = np.load(os.path.join(SHIPS_PATH, f), allow_pickle=True)
        mask = ds.item().get("label")
        image = ds.item().get("data")
        h, w, _ = image.shape
        metrics["number_pixels"].append(h*w)
        metrics["mean"].append(image.mean(axis=(0,1)))
        metrics["var"].append(image.var(axis=(0,1)))
        pixels.append(image.reshape(-1, 12))
        mask[mask>1] = 1.
        mask = mask.astype(int).squeeze(2)
        l, n = label(mask)
        N += n

        # each label is a ship
        for i in range(1, n+1):
            mask_ship = l==i
            y_ship, x_ship = mask_ship.nonzero()
            y, x = (y_ship.max()+y_ship.min()) // 2, (x_ship.max()+x_ship.min()) // 2

            bottom, top = y + image_size//2, y - image_size//2
            right, left = x + image_size//2, x - image_size//2
            if bottom > h: bottom = h
            if top < 0: top = 0
            if right > w: right = w
            if left < 0: left = 0

            image_ship = image[top:bottom, left:right]
            h_ship, w_ship, _ = image_ship.shape
            p_h = image_size - h_ship
            p_w = image_size - w_ship
            image_ship = np.pad(image_ship, ((p_h//2,p_h-p_h//2), (p_w//2,p_w-p_w//2), (0,0)), mode='symmetric')

            image_ship = np.rollaxis(image_ship, 2, 0)
            ships.append(image_ship)
            labels.append(np.array([1], dtype=bool))
        
    f_path = os.path.join(DATASET_PATH, "ships")
    ships = np.stack(ships, axis=0).astype(np.uint16)
    labels = np.stack(labels, axis=0)
    weights = np.array(metrics["number_pixels"])
    mean = (np.array(metrics["mean"]) * weights[:,None] / weights.sum()).sum(axis=0)
    var = ((np.array(metrics["var"]) + (np.array(metrics["mean"]) - mean)**2) * weights[:,None]).sum(axis=0) / weights.sum()
    pixels = np.concatenate(pixels, axis=0)
    transformer = RobustScaler(unit_variance=True, quantile_range=(5,95)).fit(pixels)
    np.savez(f_path, data=ships, labels=labels, center=transformer.center_, scale=transformer.scale_)

    print("Number of ships :", N)


def create_floating():
    for foldn in [1, 2]:
        # Storing floating objects data
        ds = FloatingSeaObjectDataset(FLOATINGOBJECTS_PATH, fold="train", foldn=foldn, output_size=image_size, hard_negative_mining=False, use_l2a_probability=0)
        images, labels = [], []
        for image, mask, info in tqdm(ds):
            image = image.astype(int)
            images.append(image)
            labels.append(np.array([0], dtype=bool))
        images = np.stack(images, axis=0).astype(np.uint16)
        labels = np.stack(labels, axis=0)

        # Scaling the data
        # Counting pixels
        N_pixels = 0
        for region in ds.regions:
            imagefile = os.path.join(FLOATINGOBJECTS_PATH, region + ".tif")
            with rio.open(imagefile) as src:
                image = src.read()
                print(image.max())
                N_pixels += image.shape[1]*image.shape[2] // 4
        # Filling pixels array
        pixels = np.empty((N_pixels, 12), dtype=np.float16)
        last = 0
        for region in tqdm(ds.regions):
            imagefile = os.path.join(FLOATINGOBJECTS_PATH, region + ".tif")
            with rio.open(imagefile) as src:
                image = src.read().astype(np.float16)
                if (image.shape[0] == 13):  # is L1C Sentinel 2 data
                    image = image[[l1cbands.index(b) for b in l2abands]]
                N_pix = image.shape[1]*image.shape[2] // 4
                image = image.reshape(12, -1).transpose()
                pixels[last:last+N_pix] = image[np.random.choice(np.arange(0, image.shape[0]), size=N_pix, replace=False)]
                last = last + N_pix
        transformer = RobustScaler(unit_variance=True, quantile_range=(5,95)).fit(pixels)
        
        # Saving the data + scaling params
        np.savez(os.path.join(DATASET_PATH, f"floatingobjects-fold{foldn}"), data=images, labels=labels, center=transformer.center_, scale=transformer.scale_)

        print("Number of floating objects :", len(ds))


if __name__ == "__main__":
    create_ships()
    create_floating()
    