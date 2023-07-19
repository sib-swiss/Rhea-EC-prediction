**Table of contents**

* [Introduction](#introduction)
* [Methods](#methods)
* [Installation](#installation)
* [Usage](#usage)
* [Troubleshooting](#troubleshooting)
* [References](#references)


## Introduction

[Rhea](https://www.rhea-db.org/) is an expert-curated knowledgebase of chemical and transport reactions of biological interest - and the standard for enzyme and transporter annotation in [UniProtKB](https://www.uniprot.org/). Rhea reactions that are also described by the [Enzyme Commission (EC)](https://en.wikipedia.org/wiki/Enzyme_Commission_number) classification are curated with the corresponding EC numbers, which are used for the hierarchical [search by EC numbers](https://www.rhea-db.org/help/search-ec-number) on the Rhea website. However, less than half of all Rhea reactions have an exact equivalent in the EC classification. Our goal here is to predict enzyme sub-sub-classes (3-level EC numbers) for Rhea reactions that are not yet covered by the EC classification to enable a hierarchical search by EC numbers across all Rhea reactions. For this purpose, we generated and tested EC number predictors for Rhea reactions, and we publish here a series of [Jupyter](https://jupyter.org/) notebooks to reproduce this work.

## Methods

The code in this repository is based on [Theia](#theia) and has been expanded with alternative methods for reaction fingerprint generation and classification to test which combination gives good results for our use case.

Our pipeline consists of the following steps:

1. Generate reaction fingerprints
   The input data are [Rhea reaction SMILES](https://ftp.expasy.org/databases/rhea/tsv/rhea-reaction-smiles.tsv). They are encoded as fingerprints with one of the the following two methods:
       a. [RXNFP](#rxnfp): learned reaction fingerprints generated based on a BERT model
       b. [DRFP](#drfp): differential reaction fingerprints generated based on circular chemical substructure extraction and hashing

2. Train a model
   The model is trained on the subset of reactions with an expert-curated EC number. This dataset is split into three subsets - training (80%), validation (10%) and test (10%) - and a model is trained with one of the the following two methods:
      a. [multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) (MLP): a feedforward artificial neural network
      b. [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) (k-NN): a supervised learning classification method

3. Predict EC numbers for the full dataset.

4. Evaluate predictions by a comparison with the [curated reaction hierarchy](https://www.rhea-db.org/help/reaction-classification).


## Installation

1. Download the appropriate version of Anaconda or Miniconda for your operating system from the [official website](https://www.anaconda.com/products/distribution) and install it by following the instructions provided on the website.

2. Install Jupyter
```
conda install jupyter
```

3. Open a terminal or command prompt and create a new conda environment with the required package installations by running the following commands:
```
conda create -n rheaec python=3.8 -y
conda activate rheaec
conda install matplotlib -y
conda install pandas -y
conda install scipy -y
conda install tqdm -y
conda install dask -y
conda install -c conda-forge scikit-learn
conda install -c conda-forge faiss
conda install -c conda-forge typer
conda install -c conda-forge rdkit
conda install -c conda-forge transformers
conda install -c anaconda networkx
conda install -c anaconda ipykernel
pip3 install torch
pip3 install drfp
pip3 install pycm

git clone https://github.com/rxn4chemistry/rxnfp.git
cd rxnfp

# Remove the line starting with 'requirements' from settings.ini
# since these dependencies are already installed and the indicated
# versions brake the environment.
sed -i '/^requirements/d' settings.ini

pip install -e .

python -m ipykernel install --user --name=rheaec
```
   
## Usage

Select the **Kernel 'rheaec'** in the Jupyter notebook menu.
Execute the Jupyter notebooks in the following order:

1. [01_encode_fingerprints.ipynb](01_encode_fingerprints.ipynb)  
   This generates the reaction fingerprints.
   Define the fingerprints type:  
   For reproducing both rxnfp and drfp results, run all cells once with 'drfp' and once with 'rxnfp' as fpencoder.
   
2. [02_train_models.ipynb](02_train_models.ipynb)  
   This trains a model.  
   Define modeltype (knn or mlp), FP_TYPE (drfp or rxnfp) and EC_LEVEL (ec1, ec12 or ec123).
    
3. [03_predict_EC.ipynb](03_predict_EC.ipynb)  
   This predicts EC numbers for the full dataset. It may take a while (~1-2h).
    
4. [04_compare_EC_predictions_to_curated_hierarchy.ipynb](04_compare_EC_predictions_to_curated_hierarchy.ipynb)  
   This evaluates the predictions by a comparison with the [curated reaction hierarchy](https://www.rhea-db.org/help/reaction-classification).


## Troubleshooting

* If bash does not find the pip command (error: "bash: pip3: command not found..."):
```
python -m ensurepip --upgrade
```

* If there is an error resulting from incorrect rust compilation on Mac (error: "can't find Rust compiler"):
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

* If dask is incorrectly installed and cannot find the compile function:
```
python -m pip install "dask[array]" --upgrade
```

## References

### Theia
Probst D  
Explainable prediction of catalysing enzymes from reactions using multilayer perceptrons.  
bioRxiv 2023.01.28.526009 (2023).  
DOI: [https://doi.org/10.1101/2023.01.28.526009](https://doi.org/10.1101/2023.01.28.526009)  
See also [GitHub](https://github.com/daenuprobst/theia).

### DRFP
Probst D, Schwaller P, Reymond JL  
Reaction classification and yield prediction using the differential reaction fingerprint DRFP.  
Digital Discovery 1:91-97(2022).  
DOI: [https://doi.org/10.1039/d1dd00006c](https://doi.org/10.1039/d1dd00006c])

### RXNFP
Schwaller P, Probst D, Vaucher A.C, Nair VH, Kreutter D, Laino T, Reymond JL  
Mapping the space of chemical reactions using attention-based neural networks.  
Nature Machine Intelligence 3:144–152(2021).  
DOI: [https://doi.org/10.1038/s42256-020-00284-w](https://doi.org/10.1038/s42256-020-00284-w)

