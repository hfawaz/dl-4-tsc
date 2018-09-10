# Deep Learning for Time Series Classification
This is the companion repository for [our paper](link/archive) titled "Deep learning for time series classification: a review". 

## Data 
The data used in this project comes from two sources: 
* The [UCR archive](http://www.cs.ucr.edu/~eamonn/time_series_data/), which contains the 85 univariate time series datasets. 
* The [MTS archive](http://www.mustafabaydogan.com/files/viewcategory/20-data-sets.html), which contains the 13 multivariate time series datasets.

## Code 
The code is divided as follows: 
* The [main.py](https://github.com/hfawaz/dl-4-tsc/blob/master/main.py) python file contains the necessary code to run an experiement. 
* The [utils](https://github.com/hfawaz/dl-4-tsc/tree/master/utils) folder contains the necessary functions to read the datasets and visualize the plots.
* The [classifiers](https://github.com/hfawaz/dl-4-tsc/tree/master/classifiers) folder contains nine python files one for each deep neural network tested in our paper. 

To run a model on one dataset you should issue the following command: 
```
python3 main.py UCR_TS_Archive_2015 Coffee fcn _itr_8
```
which means we are launching the [fcn](https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py) model on the univariate UCR archive for the Coffee dataset (see [constants.py](https://github.com/hfawaz/dl-4-tsc/blob/master/utils/constants.py) for a list of possible options).

## Prerequisites
All python packages needed are listed in [pip-requirements.txt](https://github.com/hfawaz/dl-4-tsc/blob/master/utils/pip-requirements.txt) file and can be installed simply using the pip command. 

* [numpy](http://www.numpy.org/)  
* [pandas](https://pandas.pydata.org/)  
* [sklearn](http://scikit-learn.org/stable/)  
* [scipy](https://www.scipy.org/)  
* [matplotlib](https://matplotlib.org/)  
* [tensorflow-gpu](https://www.tensorflow.org/)  
* [keras](https://keras.io/)  
* [h5py](http://docs.h5py.org/en/latest/build.html)
* [keras_contrib](https://www.github.com/keras-team/keras-contrib.git)

## Results
Our results showed that a deep residual network architecture performs best for the time series classification task. 
These results should give an insight of deep learning for TSC therefore encouraging researchers to consider the DNNs as robust classifiers for time series data. 

## Reference

If you re-use this work, please cite:

```
@article{IsmailFawaz2018deep,
  Title                    = {Deep learning for time series classification: a review},
  Author                   = {Ismail Fawaz, Hassan and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
  journal                = {ArXiv},
  Year                     = {2018}
}
```
## Acknowledgement

We would like to thank NVIDIA Corporation for the Quadro P6000 grant and the Mésocentre of Strasbourg for providing access to the cluster.
We would also like to thank François Petitjean and Charlotte Pelletier for the fruitful discussions, their feedback and comments while writing this paper.
