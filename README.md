### Installation

* Clone this repository
```
git clone https://github.com/jplumail/projet3A
cd projet3A
```

* Install the [ESA-Philab/floatingobjects](https://github.com/jplumail/floatingobjects) fork.
```
git clone https://github.com/jplumail/floatingobjects
cd floatingobjects
```

Create a new environment with conda :
```
conda env create -f environment.yml
conda activate floatingobjects
pip install -e .
```

Under the `floatingobjects/data` folder, download the data from [this Drive](https://drive.google.com/drive/folders/1QGjzRTVRQbf4YbzfUWMeIdJvYkzuipGJ). It contains the ESA floating objects database.