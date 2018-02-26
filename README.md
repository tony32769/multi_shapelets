# Multivariate Shapelets Learning

Python code for the model and experiments of the paper:
> "_Embedding the Learning of Multivariate Shapelets in a Multi-Layer Neural Network_" by Roberto Medico, Joeri Ruyssinck, Dirk Deschrijver and Tom Dhaene, Ghent University

in submission for KDD'18.

### One-Liner

A Neural Network architecture to learn meaningful multivariate shapelets for time-series classification tasks.

### Abstract

> Shapelets are discriminative subsequences extracted from time-series data. Classifiers using shapelets have proven to achieve performances competitive to state-of-the-art methods, while enhancing the model's interpretability. While a lot of research has been done for univariate time-series shapelets, extensions for the multivariate setting have not yet received much attention. To extend shapelets-based classification to a multidimensional setting, we developed a novel architecture for shapelets _learning_, by embedding them as trainable weights in a multi-layer Neural Network. We also investigated a novel initialization strategy for the shapelets, based on meaningful multidimensional motif discovery using the Matrix Profile, a recently proposed time series analysis tool. This paper describes the proposed architecture and presents results on seven publicly available benchmark datasets. Our results show how the proposed approach achieves competitive performance across the datasets, and, in contrast with the existing discovery-based methods, is applicable to larger-scale datasets. Moreover, the proposed motif-based initialization strategy helps the model convergence and performance, as well as improving interpretability of the learnt shapelets.  Finally, the shapelets learnt during training can be extracted from the model and serve as meaningful insights on the classifier's decisions and the interactions between different dimensions.


### Model

![model](https://docs.google.com/drawings/d/e/2PACX-1vTlj4Q3JO2f9B9jPZZxmFJIwoyr9_28OhAV20RGSImH-rFb5J6bQoVzkNKEd_cwhe2b3uQB0h_wn1JQ/pub?w=630&h=706)

### Requirements

- Python 2.7
- Keras with Tensorflow backend
- [mSTAMP](https://github.com/mcyeh/mstamp) (multidimensional Matrix Profile)
- Scientific computing libraries: Numpy, Pandas, Matplotlib, Sklearn

### Datasets

All the datasets used for evaluation were collected and made available to us by the authors of [A Shapelet Transform for Multivariate Time Series Classification](https://arxiv.org/abs/1712.06428) in ARFF format.

We merged dataset_TRAIN.arff and dataset_TEST.arff split (if needed), and converted the full datasets to .csv
