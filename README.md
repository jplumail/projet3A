### Installation

* Install the [ESA-Philab/floatingobjects](https://github.com/jplumail/floatingobjects) fork.
```bash
git clone https://github.com/jplumail/floatingobjects
```

Create a new environment with conda :
```bash
cd floatingobjects
conda env create -f environment.yml
conda activate floatingobjects
pip install -e . # it will install the repo as a Python package in "editable mode"
```

Under the `floatingobjects/data` folder, download the ESA floating objects database from [this Drive](https://drive.google.com/drive/folders/1QGjzRTVRQbf4YbzfUWMeIdJvYkzuipGJ).

* After that, you can clone this repository (outside of the `floatingobjects` repository)
```bash
git clone https://github.com/jplumail/projet3A
```

Under the `projet3A/models` folder, download the classifier weights from [this Drive](https://drive.google.com/drive/folders/1IJGJNVN2o7yNTA1jluq9Rd6Q3pTox8Tg).

[Follow the instructions](https://github.com/ESA-PhiLab/floatingobjects#load-pretrained-models-using-pytorch-hub) to download the U-Net and MANet pretrained weights from the ESA repo. You can put them under the `projet3A/models` folder also.