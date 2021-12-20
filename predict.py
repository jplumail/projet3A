import os
from tqdm import tqdm
import torch

from floatingobjects.predictor import PythonPredictor
from floatingobjects.data import FloatingSeaObjectDataset


if __name__ == "__main__":

    root_data = "floatingobjects\\data"
    predictions_path = "predictions"

    for model in ["unet", "manet"]:
        model_path = os.path.join(predictions_path, model)

        for fold in [1, 2]:
            fold_path = os.path.join(model_path, str(fold))
            regions = FloatingSeaObjectDataset(root_data, fold="test", foldn=fold).regions

            for classifier in ["no-classifier", "classifier"]:
                classifier_path = os.path.join(fold_path, classifier)
                os.makedirs(classifier_path, exist_ok=True)

                classifier_model_path = f"models\\checkpoint-fold{fold}.pt" if classifier=="classifier" else None
                predictor = PythonPredictor(
                    f"models\\{model}-posweight1-lr001-bs160-ep50-aug1-seed{fold-1}.pth.tar",
                    (256, 256),
                    device="cuda",
                    use_test_aug=0,
                    add_fdi_ndvi=False,
                    offset=0,
                    classifier_path=classifier_model_path
                )

                for region in tqdm(regions):
                    region_path = os.path.join(classifier_path, f"{region}.tif")
                    predictor.predict(os.path.join(root_data,f"{region}.tif"), classifier_path)
                
                del predictor
                torch.cuda.empty_cache()

