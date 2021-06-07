# HW3

## About

This is a term project of Web Intelligence and Message Understanding, NCU 2021.

## Setup

1. Install cuda.

2. Setup env and install packages.

    ``` powershell=
    conda activate wimuta_hw3_ner_env
    conda env list
    conda create --name wimuta_hw3_ner_env python=3.7
    conda activate wimuta_hw3_ner_env

    pip install tensorflow==2.3.0-rc1
    pip install tensorflow-gpu==2.3.0-rc1
    pip install sklearn
    pip install pandas
    pip install IPython
    pip install keras
    pip install tqdm
    pip install xlrd
    pip install openpyxl
    conda install -c anaconda ipython
    pip install matplotlib
    pip install tensorflow_hub
    pip install transformers==4.4.2
    pip install gin-config
    pip install tensorflow_addons

    conda install cudatoolkit
    conda install numba
    ```

3. Change package path in `conda.pth` and paste it into your `site-packages` folder of this env.
