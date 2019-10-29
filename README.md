# Update: Added the [results](https://github.com/hfawaz/dl-4-tsc/blob/master/results/results-ucr-128.csv) on the 128 datasets from the [UCR archive 2018](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).  
# Check out [our latest Inception model](https://github.com/hfawaz/InceptionTime)
# Deep Learning for Time Series Classification
This is the companion repository for [our paper](https://link.springer.com/article/10.1007%2Fs10618-019-00619-1) titled "Deep learning for time series classification: a review" published in [Data Mining and Knowledge Discovery](https://link.springer.com/journal/10618), also available on [ArXiv](https://arxiv.org/pdf/1809.04356.pdf). 

![architecture resnet](https://github.com/hfawaz/dl-4-tsc/blob/master/png/resnet-archi.png)

## Data 
The data used in this project comes from two sources: 
* The [UCR/UEA archive](http://timeseriesclassification.com/TSC.zip), which contains the 85 univariate time series datasets. 
* The [MTS archive](http://www.mustafabaydogan.com/files/viewcategory/20-data-sets.html), which contains the 13 multivariate time series datasets.

## Code 
The code is divided as follows: 
* The [main.py](https://github.com/hfawaz/dl-4-tsc/blob/master/main.py) python file contains the necessary code to run an experiement. 
* The [utils](https://github.com/hfawaz/dl-4-tsc/tree/master/utils) folder contains the necessary functions to read the datasets and visualize the plots.
* The [classifiers](https://github.com/hfawaz/dl-4-tsc/tree/master/classifiers) folder contains nine python files one for each deep neural network tested in our paper. 

To run a model on one dataset you should issue the following command: 
```
python3 main.py TSC Coffee fcn _itr_8
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
Our [results](https://github.com/hfawaz/dl-4-tsc/tree/master/results) showed that a deep residual network architecture performs best for the time series classification task. 

The following table contains the averaged accuracy over 10 runs of each implemented model on the UCR/UEA archive, with the standard deviation between parentheses. 

| Datasets                       | MLP       | FCN        | ResNet     | Encoder    | MCNN       | t-LeNet   | MCDCNN     | Time-CNN  | TWIESN     | 
|--------------------------------|-----------|------------|------------|------------|------------|-----------|------------|-----------|------------| 
| 50words                        | 68.4(7.1) | 62.7(6.1)  | 74.0(1.5)  | 72.3(1.0)  | 22.0(24.3) | 12.5(0.0) | 58.9(5.3)  | 62.1(1.0) | 49.6(2.6)  | 
| Adiac                          | 39.7(1.9) | 84.4(0.7)  | 82.9(0.6)  | 48.4(2.5)  | 2.2(0.6)   | 2.0(0.0)  | 61.0(8.7)  | 37.9(2.0) | 41.6(4.5)  | 
| ArrowHead                      | 77.8(1.2) | 84.3(1.5)  | 84.5(1.2)  | 80.4(2.9)  | 33.9(4.7)  | 30.3(0.0) | 68.5(6.7)  | 72.3(2.6) | 65.9(9.4)  | 
| Beef                           | 72.0(2.8) | 69.7(4.0)  | 75.3(4.2)  | 64.3(5.0)  | 20.0(0.0)  | 20.0(0.0) | 56.3(7.8)  | 76.3(1.1) | 53.7(14.9) | 
| BeetleFly                      | 87.0(2.6) | 86.0(9.7)  | 85.0(2.4)  | 74.5(7.6)  | 50.0(0.0)  | 50.0(0.0) | 58.0(9.2)  | 89.0(3.2) | 73.0(7.9)  | 
| BirdChicken                    | 77.5(3.5) | 95.5(3.7)  | 88.5(5.3)  | 66.5(5.8)  | 50.0(0.0)  | 50.0(0.0) | 58.0(10.3) | 60.5(9.0) | 74.0(15.6) | 
| CBF                            | 87.2(0.7) | 99.4(0.1)  | 99.5(0.3)  | 94.7(1.2)  | 33.2(0.1)  | 33.2(0.1) | 82.0(20.5) | 95.7(1.0) | 89.0(4.9)  | 
| Car                            | 76.7(2.6) | 90.5(1.4)  | 92.5(1.4)  | 75.8(2.0)  | 24.0(2.7)  | 31.7(0.0) | 73.0(3.0)  | 78.2(1.2) | 78.3(4.0)  | 
| ChlorineConcentration          | 80.2(1.1) | 81.4(0.9)  | 84.4(1.0)  | 57.3(1.1)  | 53.3(0.0)  | 53.3(0.0) | 64.3(3.8)  | 60.0(0.8) | 55.3(0.3)  | 
| CinC\_ECG\_torso                 | 84.0(1.0) | 82.4(1.2)  | 82.6(2.4)  | 91.1(2.7)  | 38.1(28.0) | 25.0(0.1) | 73.6(15.2) | 74.5(4.9) | 30.0(2.9)  | 
| Coffee                         | 99.6(1.1) | 100.0(0.0) | 100.0(0.0) | 97.9(1.8)  | 51.4(3.5)  | 53.6(0.0) | 98.2(2.5)  | 99.6(1.1) | 97.1(2.8)  | 
| Computers                      | 56.3(1.6) | 82.2(1.0)  | 81.5(1.2)  | 57.4(2.2)  | 52.2(4.8)  | 50.0(0.0) | 55.9(3.3)  | 54.8(1.5) | 62.9(4.1)  | 
| Cricket\_X                      | 59.1(1.1) | 79.2(0.7)  | 79.1(0.6)  | 69.4(1.6)  | 18.9(23.8) | 7.4(0.0)  | 49.5(5.3)  | 55.2(2.9) | 62.2(2.1)  | 
| Cricket\_Y                      | 60.0(0.8) | 78.7(1.2)  | 80.3(0.8)  | 67.5(1.0)  | 18.4(22.0) | 8.5(0.0)  | 49.7(4.3)  | 57.0(2.4) | 65.6(1.3)  | 
| Cricket\_Z                      | 61.7(0.8) | 81.1(1.0)  | 81.2(1.4)  | 69.2(1.0)  | 18.3(24.4) | 6.2(0.0)  | 49.8(3.6)  | 48.8(2.8) | 62.2(2.3)  | 
| DiatomSizeReduction            | 91.0(1.4) | 31.3(3.6)  | 30.1(0.2)  | 91.3(1.8)  | 30.1(0.7)  | 30.1(0.0) | 70.3(28.9) | 95.4(0.7) | 88.0(6.6)  | 
| DistalPhalanxOutlineAgeGroup   | 65.7(1.1) | 71.0(1.3)  | 71.7(1.3)  | 73.7(1.6)  | 46.8(0.0)  | 44.6(2.3) | 74.4(2.2)  | 75.2(1.4) | 71.0(2.1)  | 
| DistalPhalanxOutlineCorrect    | 72.6(1.3) | 76.0(1.5)  | 77.1(1.0)  | 74.1(1.4)  | 58.3(0.0)  | 58.3(0.0) | 75.3(1.8)  | 75.9(2.0) | 71.3(1.0)  | 
| DistalPhalanxTW                | 61.7(1.3) | 69.0(2.1)  | 66.5(1.6)  | 68.8(1.6)  | 30.2(0.0)  | 28.3(0.7) | 67.7(1.8)  | 67.3(2.8) | 60.9(3.0)  | 
| ECG200                         | 91.6(0.7) | 88.9(1.0)  | 87.4(1.9)  | 92.3(1.1)  | 64.0(0.0)  | 64.0(0.0) | 83.3(3.9)  | 81.4(1.3) | 84.2(5.1)  | 
| ECG5000                        | 92.9(0.1) | 94.0(0.1)  | 93.4(0.2)  | 94.0(0.2)  | 61.8(10.9) | 58.4(0.0) | 93.7(0.6)  | 92.8(0.2) | 91.9(0.2)  | 
| ECGFiveDays                    | 97.0(0.5) | 98.7(0.3)  | 97.5(1.9)  | 98.2(0.7)  | 49.9(0.3)  | 49.7(0.0) | 76.2(13.4) | 88.2(1.8) | 69.8(14.1) | 
| Earthquakes                    | 71.7(1.3) | 72.7(1.7)  | 71.2(2.0)  | 74.8(0.7)  | 74.8(0.0)  | 74.8(0.0) | 74.9(0.2)  | 70.0(1.9) | 74.8(0.0)  | 
| ElectricDevices                | 59.2(1.1) | 70.2(1.2)  | 72.9(0.9)  | 67.4(1.1)  | 33.6(19.8) | 24.2(0.0) | 64.4(1.2)  | 68.1(1.0) | 60.7(0.7)  | 
| FISH                           | 84.8(0.8) | 95.8(0.6)  | 97.9(0.8)  | 86.6(0.9)  | 13.4(1.3)  | 12.6(0.0) | 75.8(3.9)  | 84.9(0.5) | 87.5(3.4)  | 
| FaceAll                        | 79.3(1.1) | 94.5(0.9)  | 83.9(2.0)  | 79.3(0.8)  | 17.0(19.5) | 8.0(0.0)  | 71.7(2.3)  | 76.8(1.1) | 65.7(2.5)  | 
| FaceFour                       | 84.0(1.4) | 92.8(0.9)  | 95.5(0.0)  | 81.5(2.6)  | 26.8(5.7)  | 29.5(0.0) | 71.2(13.5) | 90.6(1.1) | 85.5(6.2)  | 
| FacesUCR                       | 83.3(0.3) | 94.6(0.2)  | 95.5(0.4)  | 87.4(0.4)  | 15.3(2.7)  | 14.3(0.0) | 75.6(5.1)  | 86.9(0.7) | 64.4(2.0)  | 
| FordA                          | 73.0(0.4) | 90.4(0.2)  | 92.0(0.4)  | 92.3(0.3)  | 51.3(0.0)  | 51.0(0.8) | 79.5(2.6)  | 88.1(0.7) | 52.8(2.1)  | 
| FordB                          | 60.3(0.3) | 87.8(0.6)  | 91.3(0.3)  | 89.0(0.5)  | 49.8(1.2)  | 51.2(0.0) | 53.3(2.9)  | 80.6(1.5) | 50.3(1.2)  | 
| Gun\_Point                      | 92.7(1.1) | 100.0(0.0) | 99.1(0.7)  | 93.6(3.2)  | 51.3(3.9)  | 49.3(0.0) | 86.7(9.6)  | 93.2(1.9) | 96.1(2.3)  | 
| Ham                            | 69.1(1.4) | 71.8(1.4)  | 75.7(2.7)  | 72.7(1.2)  | 50.6(1.4)  | 51.4(0.0) | 73.3(4.2)  | 71.1(2.0) | 72.3(6.3)  | 
| HandOutlines                   | 91.8(0.5) | 80.6(7.9)  | 91.1(1.4)  | 89.9(2.3)  | 64.1(0.0)  | 64.1(0.0) | 90.9(0.6)  | 88.8(1.2) | 66.0(0.7)  | 
| Haptics                        | 43.3(1.4) | 48.0(2.4)  | 51.9(1.2)  | 42.7(1.6)  | 20.9(3.5)  | 20.8(0.0) | 40.4(3.3)  | 36.6(2.4) | 40.4(4.5)  | 
| Herring                        | 52.8(3.9) | 60.8(7.7)  | 61.9(3.8)  | 58.6(4.8)  | 59.4(0.0)  | 59.4(0.0) | 60.0(5.2)  | 53.9(1.7) | 59.1(6.5)  | 
| InlineSkate                    | 33.7(1.0) | 33.9(0.8)  | 37.3(0.9)  | 29.2(0.9)  | 16.7(1.6)  | 16.5(1.1) | 21.5(2.2)  | 28.7(1.2) | 33.0(6.8)  | 
| InsectWingbeatSound            | 60.7(0.4) | 39.3(0.6)  | 50.7(0.9)  | 63.3(0.6)  | 15.8(14.2) | 9.1(0.0)  | 58.3(2.6)  | 58.3(0.6) | 43.7(2.0)  | 
| ItalyPowerDemand               | 95.4(0.2) | 96.1(0.3)  | 96.3(0.4)  | 96.5(0.5)  | 50.0(0.2)  | 49.9(0.0) | 95.5(1.9)  | 95.5(0.4) | 88.0(2.2)  | 
| LargeKitchenAppliances         | 47.3(0.6) | 90.2(0.4)  | 90.0(0.5)  | 61.9(2.6)  | 41.0(16.5) | 33.3(0.0) | 43.4(2.8)  | 66.6(5.0) | 77.9(1.8)  | 
| Lighting2                      | 67.0(2.1) | 73.9(1.4)  | 77.0(1.7)  | 69.2(4.6)  | 55.7(5.2)  | 54.1(0.0) | 63.0(5.9)  | 63.6(2.5) | 70.3(4.1)  | 
| Lighting7                      | 63.0(1.7) | 82.7(2.3)  | 84.5(2.0)  | 62.5(2.3)  | 31.0(11.3) | 26.0(0.0) | 53.4(5.9)  | 65.1(3.3) | 66.4(6.6)  | 
| MALLAT                         | 91.8(0.6) | 96.7(0.9)  | 97.2(0.3)  | 87.6(2.0)  | 13.5(3.7)  | 12.3(0.1) | 90.1(5.7)  | 92.0(0.7) | 59.6(9.8)  | 
| Meat                           | 89.7(1.7) | 85.3(6.9)  | 96.8(2.5)  | 74.2(11.0) | 33.3(0.0)  | 33.3(0.0) | 70.5(8.8)  | 90.2(1.8) | 96.8(2.0)  | 
| MedicalImages                  | 72.1(0.7) | 77.9(0.4)  | 77.0(0.7)  | 73.4(1.5)  | 51.4(0.0)  | 51.4(0.0) | 64.0(1.4)  | 67.6(1.1) | 64.9(2.7)  | 
| MiddlePhalanxOutlineAgeGroup   | 53.1(1.8) | 55.3(1.8)  | 56.9(2.1)  | 57.9(2.9)  | 18.8(0.0)  | 57.1(0.0) | 58.5(3.8)  | 56.6(1.5) | 58.1(2.6)  | 
| MiddlePhalanxOutlineCorrect    | 77.0(1.1) | 80.1(1.0)  | 80.9(1.2)  | 76.1(2.3)  | 57.0(0.0)  | 57.0(0.0) | 81.1(1.6)  | 76.6(1.3) | 74.4(2.3)  | 
| MiddlePhalanxTW                | 53.4(1.6) | 51.2(1.8)  | 48.4(2.0)  | 59.2(1.0)  | 27.3(0.0)  | 28.6(0.0) | 58.1(2.4)  | 54.9(1.7) | 53.9(2.9)  | 
| MoteStrain                     | 85.8(0.9) | 93.7(0.5)  | 92.8(0.5)  | 84.0(1.0)  | 50.8(4.0)  | 53.9(0.0) | 76.5(14.4) | 88.2(0.9) | 78.5(4.2)  | 
| NonInvasiveFatalECG\_Thorax1    | 91.6(0.4) | 95.6(0.3)  | 94.5(0.3)  | 91.6(0.4)  | 16.1(29.3) | 2.9(0.0)  | 90.5(1.2)  | 86.5(0.5) | 49.4(4.2)  | 
| NonInvasiveFatalECG\_Thorax2    | 91.7(0.3) | 95.3(0.3)  | 94.6(0.3)  | 93.2(0.9)  | 16.0(29.2) | 2.9(0.0)  | 91.5(1.5)  | 89.8(0.3) | 52.5(3.2)  | 
| OSULeaf                        | 55.7(1.0) | 97.7(0.9)  | 97.9(0.8)  | 57.6(2.0)  | 24.3(12.8) | 18.2(0.0) | 37.8(4.6)  | 46.2(2.7) | 59.5(5.4)  | 
| OliveOil                       | 66.7(3.8) | 72.3(16.6) | 83.0(8.5)  | 40.0(0.0)  | 38.0(4.2)  | 38.0(4.2) | 40.0(0.0)  | 40.0(0.0) | 79.0(6.1)  | 
| PhalangesOutlinesCorrect       | 73.5(2.1) | 82.0(0.5)  | 83.9(1.2)  | 76.7(1.4)  | 61.3(0.0)  | 61.3(0.0) | 80.3(1.1)  | 77.1(4.7) | 65.4(0.4)  | 
| Phoneme                        | 9.6(0.3)  | 32.5(0.5)  | 33.4(0.7)  | 17.2(0.8)  | 13.2(4.0)  | 11.3(0.0) | 13.0(1.0)  | 9.5(0.3)  | 12.8(1.4)  | 
| Plane                          | 97.8(0.5) | 100.0(0.0) | 100.0(0.0) | 97.6(0.8)  | 13.0(4.5)  | 13.4(1.4) | 96.5(3.2)  | 96.5(1.4) | 100.0(0.0) | 
| ProximalPhalanxOutlineAgeGroup | 85.6(0.5) | 83.1(1.3)  | 85.3(0.8)  | 84.4(1.3)  | 48.8(0.0)  | 48.8(0.0) | 83.8(0.8)  | 82.8(1.6) | 84.4(0.5)  | 
| ProximalPhalanxOutlineCorrect  | 73.3(1.8) | 90.3(0.7)  | 92.1(0.6)  | 79.1(1.8)  | 68.4(0.0)  | 68.4(0.0) | 87.3(1.8)  | 81.2(2.6) | 82.1(0.9)  | 
| ProximalPhalanxTW              | 76.7(0.7) | 76.7(0.9)  | 78.0(1.7)  | 81.2(1.1)  | 35.1(0.0)  | 34.6(1.0) | 79.7(1.3)  | 78.3(1.2) | 78.1(0.7)  | 
| RefrigerationDevices           | 37.9(2.1) | 50.8(1.0)  | 52.5(2.5)  | 48.8(1.9)  | 33.3(0.0)  | 33.3(0.0) | 36.9(3.8)  | 43.9(1.0) | 50.1(1.5)  | 
| ScreenType                     | 40.3(1.0) | 62.5(1.6)  | 62.2(1.4)  | 38.3(2.2)  | 34.1(2.4)  | 33.3(0.0) | 42.7(1.8)  | 38.9(0.9) | 43.1(4.7)  | 
| ShapeletSim                    | 50.3(3.1) | 72.4(5.6)  | 77.9(15.0) | 53.0(4.7)  | 50.0(0.0)  | 50.0(0.0) | 50.7(4.1)  | 50.0(1.3) | 61.7(10.2) | 
| ShapesAll                      | 77.1(0.5) | 89.5(0.4)  | 92.1(0.4)  | 75.8(0.9)  | 13.2(24.3) | 1.7(0.0)  | 61.3(5.3)  | 61.9(0.9) | 62.9(2.6)  | 
| SmallKitchenAppliances         | 37.1(1.9) | 78.3(1.3)  | 78.6(0.8)  | 59.6(1.8)  | 36.9(11.3) | 33.3(0.0) | 48.5(3.6)  | 61.5(2.7) | 65.6(1.9)  | 
| SonyAIBORobotSurface           | 67.2(1.3) | 96.0(0.7)  | 95.8(1.3)  | 74.3(1.9)  | 44.3(4.5)  | 42.9(0.0) | 65.3(10.9) | 68.7(2.3) | 63.8(9.9)  | 
| SonyAIBORobotSurfaceII         | 83.4(0.7) | 97.9(0.5)  | 97.8(0.5)  | 83.9(1.0)  | 59.4(7.4)  | 61.7(0.0) | 77.4(6.7)  | 84.1(1.7) | 69.7(4.3)  | 
| StarLightCurves                | 94.9(0.2) | 96.1(0.9)  | 97.2(0.3)  | 95.7(0.5)  | 65.4(16.1) | 57.7(0.0) | 93.9(1.2)  | 92.6(0.2) | 85.0(0.2)  | 
| Strawberry                     | 96.1(0.5) | 97.2(0.3)  | 98.1(0.4)  | 94.6(0.9)  | 64.3(0.0)  | 64.3(0.0) | 95.6(0.6)  | 95.9(0.3) | 89.5(2.0)  | 
| SwedishLeaf                    | 85.1(0.5) | 96.9(0.5)  | 95.6(0.4)  | 93.0(1.1)  | 11.8(13.2) | 6.5(0.4)  | 84.6(3.6)  | 88.4(1.1) | 82.5(1.4)  | 
| Symbols                        | 83.2(1.0) | 95.5(1.0)  | 90.6(2.3)  | 82.1(1.9)  | 22.6(16.9) | 17.4(0.0) | 75.6(11.5) | 81.0(0.7) | 75.0(8.8)  | 
| ToeSegmentation1               | 58.3(0.9) | 96.1(0.5)  | 96.3(0.6)  | 65.9(2.6)  | 50.5(2.7)  | 52.6(0.0) | 49.0(2.5)  | 59.5(2.2) | 86.5(3.2)  | 
| ToeSegmentation2               | 74.5(1.9) | 88.0(3.3)  | 90.6(1.7)  | 79.5(2.8)  | 63.2(30.9) | 81.5(0.0) | 44.3(15.2) | 73.8(2.8) | 84.2(4.6)  | 
| Trace                          | 80.7(0.7) | 100.0(0.0) | 100.0(0.0) | 96.0(1.8)  | 35.4(27.7) | 24.0(0.0) | 86.3(5.4)  | 95.0(2.5) | 95.9(1.9)  | 
| TwoLeadECG                     | 76.2(1.3) | 100.0(0.0) | 100.0(0.0) | 86.3(2.6)  | 50.0(0.0)  | 50.0(0.0) | 76.0(16.8) | 87.2(2.1) | 85.2(11.5) | 
| Two\_Patterns                   | 94.6(0.3) | 87.1(0.3)  | 100.0(0.0) | 100.0(0.0) | 40.3(31.1) | 25.9(0.0) | 97.8(0.6)  | 99.2(0.3) | 87.1(1.1)  | 
| UWaveGestureLibraryAll         | 95.5(0.2) | 81.7(0.3)  | 86.0(0.4)  | 95.4(0.1)  | 28.9(34.7) | 12.8(0.2) | 92.9(1.1)  | 91.8(0.4) | 55.6(2.5)  | 
| Wine                           | 56.5(7.1) | 58.7(8.3)  | 74.4(8.5)  | 50.0(0.0)  | 50.0(0.0)  | 50.0(0.0) | 50.0(0.0)  | 51.7(5.1) | 75.9(9.1)  | 
| WordsSynonyms                  | 59.8(0.8) | 56.4(1.2)  | 62.2(1.5)  | 61.3(0.9)  | 28.4(13.6) | 21.9(0.0) | 46.3(6.1)  | 56.6(0.8) | 49.0(3.0)  | 
| Worms                          | 45.7(2.4) | 76.5(2.2)  | 79.1(2.5)  | 57.1(3.7)  | 42.9(0.0)  | 42.9(0.0) | 42.6(5.5)  | 38.3(2.5) | 46.6(4.5)  | 
| WormsTwoClass                  | 60.1(1.5) | 72.6(2.7)  | 74.7(3.3)  | 63.9(4.4)  | 57.1(0.0)  | 55.7(4.5) | 57.0(1.9)  | 53.8(2.6) | 57.0(2.3)  | 
| synthetic\_control              | 97.6(0.4) | 98.5(0.3)  | 99.8(0.2)  | 99.6(0.3)  | 29.8(27.8) | 16.7(0.0) | 98.3(1.2)  | 99.0(0.4) | 87.4(1.6)  | 
| uWaveGestureLibrary\_X          | 76.7(0.3) | 75.4(0.4)  | 78.0(0.4)  | 78.6(0.4)  | 18.9(21.3) | 12.5(0.4) | 71.1(1.5)  | 71.1(1.1) | 60.6(1.5)  | 
| uWaveGestureLibrary\_Y          | 69.8(0.2) | 63.9(0.6)  | 67.0(0.7)  | 69.6(0.6)  | 23.7(24.0) | 12.1(0.0) | 63.6(1.2)  | 62.6(0.7) | 52.0(2.1)  | 
| uWaveGestureLibrary\_Z          | 69.7(0.2) | 72.6(0.5)  | 75.0(0.4)  | 71.1(0.5)  | 18.0(18.4) | 12.1(0.0) | 65.0(1.8)  | 64.2(0.9) | 56.5(2.0)  | 
| wafer                          | 99.6(0.0) | 99.7(0.0)  | 99.9(0.1)  | 99.6(0.0)  | 91.3(4.4)  | 89.2(0.0) | 99.2(0.3)  | 96.1(0.1) | 91.4(0.5)  | 
| yoga                           | 85.5(0.4) | 83.9(0.7)  | 87.0(0.9)  | 82.0(0.6)  | 53.6(0.0)  | 53.6(0.0) | 76.2(3.9)  | 78.1(0.7) | 60.7(1.9)  | 
| **Average\_Rank**               | 4.611765  | 2.682353   | 1.994118   | 3.682353   | 8.017647   | 8.417647  | 5.376471   | 4.970588  | 5.247059   | 
| **Wins**                       | 4         | 18         | 41         | 10         | 0          | 0         | 3          | 4         | 1          | 

The following table contains the averaged accuracy over 10 runs of each implemented model on the MTS archive, with the standard deviation between parentheses. 

| Datasets              | MLP        | FCN        | ResNet     | Encoder    | MCNN      | t-LeNet    | MCDCNN     | Time-CNN   | TWIESN     | 
|-----------------------|------------|------------|------------|------------|-----------|------------|------------|------------|------------| 
| AUSLAN                | 93.3(0.5)  | 97.5(0.4)  | 97.4(0.3)  | 93.8(0.5)  | 1.1(0.0)  | 1.1(0.0)   | 85.4(2.7)  | 72.6(3.5)  | 72.4(1.6)  | 
| ArabicDigits          | 96.9(0.2)  | 99.4(0.1)  | 99.6(0.1)  | 98.1(0.1)  | 10.0(0.0) | 10.0(0.0)  | 95.9(0.2)  | 95.8(0.3)  | 85.3(1.4)  | 
| CMUsubject16          | 60.0(16.9) | 100.0(0.0) | 99.7(1.1)  | 98.3(2.4)  | 53.1(4.4) | 51.0(5.3)  | 51.4(5.0)  | 97.6(1.7)  | 89.3(6.8)  | 
| CharacterTrajectories | 96.9(0.2)  | 99.0(0.1)  | 99.0(0.2)  | 97.1(0.2)  | 5.4(0.8)  | 6.7(0.0)   | 93.8(1.7)  | 96.0(0.8)  | 92.0(1.3)  | 
| ECG                   | 74.8(16.2) | 87.2(1.2)  | 86.7(1.3)  | 87.2(0.8)  | 67.0(0.0) | 67.0(0.0)  | 50.0(17.9) | 84.1(1.7)  | 73.7(2.3)  | 
| JapaneseVowels        | 97.6(0.2)  | 99.3(0.2)  | 99.2(0.3)  | 97.6(0.6)  | 9.2(2.5)  | 23.8(0.0)  | 94.4(1.4)  | 95.6(1.0)  | 96.5(0.7)  | 
| KickvsPunch           | 61.0(12.9) | 54.0(13.5) | 51.0(8.8)  | 61.0(9.9)  | 54.0(9.7) | 50.0(10.5) | 56.0(8.4)  | 62.0(6.3)  | 67.0(14.2) | 
| Libras                | 78.0(1.0)  | 96.4(0.7)  | 95.4(1.1)  | 78.3(0.9)  | 6.7(0.0)  | 6.7(0.0)   | 65.1(3.9)  | 63.7(3.3)  | 79.4(1.3)  | 
| NetFlow               | 55.0(26.1) | 89.1(0.4)  | 62.7(23.4) | 77.7(0.5)  | 77.9(0.0) | 72.3(17.6) | 63.0(18.2) | 89.0(0.9)  | 94.5(0.4)  | 
| UWave                 | 90.1(0.3)  | 93.4(0.3)  | 92.6(0.4)  | 90.8(0.4)  | 12.5(0.0) | 12.5(0.0)  | 84.5(1.6)  | 85.9(0.7)  | 75.4(6.3)  | 
| Wafer                 | 89.4(0.0)  | 98.2(0.5)  | 98.9(0.4)  | 98.6(0.2)  | 89.4(0.0) | 89.4(0.0)  | 65.8(38.1) | 94.8(2.1)  | 94.9(0.6)  | 
| WalkvsRun             | 70.0(15.8) | 100.0(0.0) | 100.0(0.0) | 100.0(0.0) | 75.0(0.0) | 60.0(24.2) | 45.0(25.8) | 100.0(0.0) | 94.4(9.1)  | 
| **Average\_Rank**          | 5.208333   | 2.000000   | 2.875000   | 3.041667   | 7.583333  | 8.000000   | 6.833333   | 4.625000   | 4.833333   | 
| **Wins**                  | 0          | 5          | 3          | 0          | 0         | 0          | 0          | 0          | 2          | 

These results should give an insight of deep learning for TSC therefore encouraging researchers to consider the DNNs as robust classifiers for time series data. 

## Reference

If you re-use this work, please cite:

```
@article{IsmailFawaz2018deep,
  Title                    = {Deep learning for time series classification: a review},
  Author                   = {Ismail Fawaz, Hassan and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
  journal                  = {Data Mining and Knowledge Discovery},
  Year                     = {2019},
  volume                   = {33},
  number                   = {4},
  pages                    = {917--963},
}
```
## Acknowledgement

We would like to thank the providers of the [UCR/UEA archive](http://timeseriesclassification.com/TSC.zip). 
We would also like to thank NVIDIA Corporation for the Quadro P6000 grant and the Mésocentre of Strasbourg for providing access to the cluster.
We would also like to thank François Petitjean and Charlotte Pelletier for the fruitful discussions, their feedback and comments while writing this paper.
