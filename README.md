# DGraphDTA
A method for predicting the affinity of drug-protein based on graph neural network, which is called DGraphDTA (double graph DTA predictor). The method can predict the affinity only using the molecule SMILES and protein sequence. This repo gits from GraphDTA, and compared with GraphDTA, the method constructs both the graph of protein and small molecule at the same time. The protein graph is constructed according to contact map.

<div align=center><img width="900" height="400" src="https://github.com/595693085/DGraphDTA/blob/master/figures/architecture.png"/></div>

## dependencies
numpy == 1.17.4 <br>
kreas == 2.3.1 <br>
Pconsc4 == 0.4 <br>
pytorch == 1.1.0 <br>
PyG (torch-geometric) == 1.3.2 <br>
hhsuite <br>
rdkit <br>
ccmpred <br>

## train
1. Prepare the data. Get all msa files of the proteins in datasets (for more detail description of datasets, please refer to [datasets](https://github.com/hkmztrk/DeepDTA/blob/master/data/README.md)), and using Pconsc4 to predict all the contact map. A script in the repo can be run to do all the steps: <br>
**python scripts.py** <br>
Before runing the script, please edit the alnFilePrepare() function to indicate all program paths (A eaxmple is shown in it). 

2. Run the training code. <br>
**python training.py 0 0 0** <br>
where the three parameters are dataset selection, gpu selection and fold selection (for 5-fold training and validation).

## test
Mast run after the train step. This step is to reproduce the experiments. <br>
**python test.py 0 0 0** <br>
and the the three parameters are same to the training.

