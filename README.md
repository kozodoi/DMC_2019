# DMC_2019

if you create a new virtual environment
use:

```
conda create -n dmc python=3.6
conda activate dmc
```

and then

```
conda install -n dmc --yes --file requirements.txt
pip install lightgbm
pip install imblearn
```

or if you install the packages  in your base environment

```
conda install --yes --file requirements.txt
pip install lightgbm
pip install imblearn
```
