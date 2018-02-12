# Multivariate Shapelets Learning

Python code for the model and experiments of the paper:
> "_Embedding the Learning of Multivariate Shapelets in a Multi-Layer Neural Network_" by Roberto Medico, Joeri Ruyssinck, Dirk Deschrijver and Tom Dhaene, Ghent University

in submission for KDD'18.

### Requirements

- Python 2.7
- Keras with Tensorflow backend
- [mSTAMP](https://github.com/mcyeh/mstamp) (multidimensional Matrix Profile)
- Scientific computing libraries: Numpy, Pandas, Matplotlib, Sklearn

### Datasets

All the datasets used for evaluation were collected and made available to us by the authors of [A Shapelet Transform for Multivariate Time Series Classification](https://arxiv.org/abs/1712.06428) in ARFF format.

We merged dataset_TRAIN.arff and dataset_TEST.arff split (if needed), and converted the full datasets to .csv
