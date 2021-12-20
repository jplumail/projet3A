### Installation

* Clone this repository
```
git clone https://github.com/jplumail/projet3A
cd projet3A
```

* Install the [ESA-Philab/floatingobjects](https://github.com/jplumail/floatingobjects) fork.
```
git clone https://github.com/jplumail/floatingobjects
```

Create a new environment with conda :
```
cd floatingobjects
conda env create -f environment.yml
conda activate floatingobjects
pip install -e .
```

Under the `floatingobjects/data` folder, download the data from [this Drive](https://drive.google.com/drive/folders/1QGjzRTVRQbf4YbzfUWMeIdJvYkzuipGJ). It contains the ESA floating objects database.

Under the `projet3A/models` folder, download the classifier weights from [this Drive](https://drive.google.com/drive/folders/1IJGJNVN2o7yNTA1jluq9Rd6Q3pTox8Tg).

[Follow the instructions](https://github.com/ESA-PhiLab/floatingobjects#load-pretrained-models-using-pytorch-hub) to download the U-Net and MANet pretrained weights from the ESA repo.