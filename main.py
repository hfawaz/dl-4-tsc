from utils.utils import generate_results_csv
from utils.utils import transform_labels
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam

import numpy as np
import sys
import sklearn 

def fit_classifier(): 
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train,y_test),axis =0)))

    # make the min to zero of labels
    y_train,y_test = transform_labels(y_train,y_test)

    # save orignal y because later we will use binary
    y_true = y_test.astype(np.int64) 
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train,y_test),axis =0).reshape(-1,1))
    y_train = enc.transform(y_train.reshape(-1,1)).toarray()
    y_test = enc.transform(y_test.reshape(-1,1)).toarray()

    if len(x_train.shape) == 2: # if univariate 
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
        x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name,input_shape, nb_classes, output_directory)

    classifier.fit(x_train,y_train,x_test,y_test, y_true)

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose = False):
    if classifier_name=='fcn': 
        from classifiers import fcn        
        return fcn.Classifier_FCN(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='mlp':
        from  classifiers import  mlp 
        return mlp.Classifier_MLP(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='resnet':
        from  classifiers import resnet 
        return resnet.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='mcnn':
        from  classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory,verbose)
    if classifier_name=='tlenet':
        from  classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory,verbose)
    if classifier_name=='twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory,verbose)
    if classifier_name=='encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='cnn': # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory,input_shape, nb_classes, verbose)

############################################### main 

# change this directory for your machine
# it should contain the archive folder containing both univariate and multivariate archives
root_dir = '/mnt/nfs/casimir/'

if sys.argv[1]=='transform_mts_to_ucr_format':
    transform_mts_to_ucr_format()
elif sys.argv[1]=='visualize_filter':
    visualize_filter(root_dir)
elif sys.argv[1]=='viz_for_survey_paper':
    viz_for_survey_paper(root_dir)
elif sys.argv[1]=='viz_cam':
    viz_cam(root_dir)
elif sys.argv[1]=='generate_results_csv':
    res = generate_results_csv('results.csv',root_dir)
    print(res)
else:
    # this is the code used to launch an experiment on a dataset
    archive_name = sys.argv[1]
    dataset_name = sys.argv[2]
    classifier_name=sys.argv[3]
    itr = sys.argv[4]

    if itr == '_itr_0': 
        itr = ''

    output_directory = root_dir+'/results/'+classifier_name+'/'+archive_name+itr+'/'+\
        dataset_name+'/'

    output_directory = create_directory(output_directory)

    print('Method: ',archive_name, dataset_name, classifier_name, itr)

    if output_directory is None: 
        print('Already done')
    else: 

        datasets_dict = read_dataset(root_dir,archive_name,dataset_name)

        fit_classifier()

        print('DONE')

        # the creation of this directory means
        create_directory(output_directory+'/DONE')
