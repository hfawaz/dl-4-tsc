# model twi-esn 
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as slinalg

from sklearn.linear_model import Ridge

import gc

from utils.utils import calculate_metrics
from utils.utils import create_directory
from sklearn.model_selection import train_test_split

import time

class Classifier_TWIESN:

	def __init__(self, output_directory,verbose): 
		self.output_directory = output_directory
		self.verbose = verbose
		# hyperparameters 
		first_config = {'N_x':250,'connect':0.5,'scaleW_in':1.0,'lamda':0.0}
		second_config = {'N_x':250,'connect':0.5,'scaleW_in':2.0,'lamda':0.05}
		third_config = {'N_x':500,'connect':0.1,'scaleW_in':2.0,'lamda':0.05}
		fourth_config = {'N_x':800,'connect':0.1,'scaleW_in':2.0,'lamda':0.05}
		self.configs = [first_config,second_config,third_config,fourth_config]
		self.rho_s = [0.55,0.9,2.0,5.0]
		self.alpha = 0.1 # leaky rate


	def init_matrices(self):
		self.W_in = (2.0*np.random.rand(self.N_x,self.num_dim)-1.0)/(2.0*self.scaleW_in)

		converged = False

		i =0 

		# repeat because could not converge to find eigenvalues 
		while(not converged):
			i+=1

			# generate sparse, uniformly distributed weights
			self.W = sparse.rand(self.N_x,self.N_x,density=self.connect).todense()

			# ensure that the non-zero values are uniformly distributed 
			self.W[np.where(self.W>0)] -= 0.5

			try:
				# get the largest eigenvalue 
				eig, _ = slinalg.eigs(self.W,k=1,which='LM')
				converged = True
			except: 
				print('not converged ',i)
				continue

		# adjust the spectral radius
		self.W /= np.abs(eig)/self.rho

	def compute_state_matrix(self, x_in):
		# number of instances 
		n = x_in.shape[0]
		# the state matrix to be computed
		X_t = np.zeros((n, self.T, self.N_x),dtype=np.float64) 
		# previous state matrix
		X_t_1 = np.zeros((n, self.N_x),dtype=np.float64) 
		# loop through each time step 
		for t in range(self.T):
			# get all the time series data points for the time step t  
			curr_in = x_in[:,t,:]
			# calculate the linear activation 
			curr_state = np.tanh(self.W_in.dot(curr_in.T)+self.W.dot(X_t_1.T)).T
			# apply leakage 
			curr_state = (1-self.alpha)*X_t_1 + self.alpha*curr_state
			# save in previous state 
			X_t_1 = curr_state
			# save in state matrix 
			X_t[:,t,:] = curr_state
				
		return X_t

	def reshape_prediction(self,y_pred, num_instances,length_series): 
		# reshape so the first axis has the number of instances 
		new_y_pred = y_pred.reshape(num_instances,length_series,y_pred.shape[-1])
		# average the predictions of instances 
		new_y_pred = np.average(new_y_pred, axis=1)
		# get the label with maximum prediction over the last label axis 
		new_y_pred = np.argmax(new_y_pred,axis=1)
		return new_y_pred

	def train(self):
		start_time = time.time()

		################
		### Training ###
		################

		# init the matrices 
		self.init_matrices()
		# compute the state matrices which is the new feature space  
		state_matrix = self.compute_state_matrix(self.x_train)
		# add the input to form the new feature space and transform to 
		# the new feature space to be feeded to the classifier 
		new_x_train = np.concatenate((self.x_train,state_matrix), axis=2).reshape(
			self.N * self.T , self.num_dim+self.N_x)
		# memory free 
		state_matrix = None 
		gc.collect()
		# transform the corresponding labels
		new_labels = np.repeat(self.y_train,self.T,axis=0)
		# new model 
		ridge_classifier = Ridge(alpha=self.lamda)
		# fit the new feature space 
		ridge_classifier.fit(new_x_train,new_labels)

		################
		## Validation ##
		################
		# compute state matrix for validation set 
		state_matrix = self.compute_state_matrix(self.x_val)
		# add the input to form the new feature space and transform to
		# the new feature space to be feeded to the classifier 
		new_x_val = np.concatenate((self.x_val,state_matrix), axis=2).reshape(
			self.x_val.shape[0] * self.T , self.num_dim+self.N_x)
		# get the prediction on the train set 
		y_pred_val = ridge_classifier.predict(new_x_val)
		# reconstruct the training prediction 
		y_pred_val = self.reshape_prediction(y_pred_val,self.x_val.shape[0],self.T)
		# get the metrics for the train 
		df_val_metrics = calculate_metrics(np.argmax(self.y_val, axis=1),y_pred_val,0.0)
		# get the train accuracy
		train_acc = df_val_metrics['accuracy'][0]

		###############
		### Testing ###
		###############

		# get the predicition on the test set
		# transform the test set to the new features  
		state_matrix = self.compute_state_matrix(self.x_test)
		# add the input to form the new feature space and transform to the new feature space to be feeded to the classifier 
		new_x_test = np.concatenate((self.x_test,state_matrix), axis=2).reshape(self.x_test.shape[0] * self.T , self.num_dim+self.N_x)
		# memory free 
		state_matrix = None 
		gc.collect()
		# get the prediction on the test set 
		y_pred = ridge_classifier.predict(new_x_test)
		# reconstruct the test predictions 
		y_pred = self.reshape_prediction(y_pred,self.x_test.shape[0],self.T)

		duration = time.time() - start_time
		# get the metrics for the test predictions 
		df_metrics = calculate_metrics(self.y_true,y_pred,duration)

		# get the output layer weights 
		self.W_out = ridge_classifier.coef_
		ridge_classifier = None 
		gc.collect()
		# save the model 
		np.savetxt(self.output_directory+'W_in.txt', self.W_in)
		np.savetxt(self.output_directory+'W.txt', self.W)
		np.savetxt(self.output_directory+'W_out.txt', self.W_out)

		# save the metrics 
		df_metrics.to_csv(self.output_directory+'df_metrics.csv', index=False)

		# return the training accuracy and the prediction metrics on the test set
		return df_metrics , train_acc


	def fit(self,x_train,y_train,x_test,y_test, y_true):
		best_train_acc = -1

		self.num_dim = x_train.shape[2]
		self.T = x_train.shape[1]
		self.x_test = x_test
		self.y_true = y_true
		self.y_test = y_test

		# split train to validation set to choose best hyper parameters 
		self.x_train, self.x_val, self.y_train,self.y_val = \
			train_test_split(x_train,y_train, test_size=0.2)
		self.N = self.x_train.shape[0]

		# limit the hyperparameter search if dataset is too big 
		if self.x_train.shape[0] > 1000 or self.x_test.shape[0] > 1000 : 
			for config in self.configs: 
				config['N_x'] = 100
			self.configs = [self.configs[0],self.configs[1],self.configs[2]]

		output_directory_root = self.output_directory 
		# grid search 
		for idx_config in range(len(self.configs)): 
			for rho in self.rho_s:
				self.rho = rho 
				self.N_x = self.configs[idx_config]['N_x']
				self.connect = self.configs[idx_config]['connect']
				self.scaleW_in = self.configs[idx_config]['scaleW_in']
				self.lamda = self.configs[idx_config]['lamda']
				self.output_directory = output_directory_root+'/hyper_param_search/'+\
					'/config_'+str(idx_config)+'/'+'rho_'+str(rho)+'/'
				create_directory(self.output_directory)
				df_metrics, train_acc = self.train() 

				if best_train_acc < train_acc : 
					best_train_acc = train_acc
					df_metrics.to_csv(output_directory_root+'df_metrics.csv', index=False)
					np.savetxt(output_directory_root+'W_in.txt', self.W_in)
					np.savetxt(output_directory_root+'W.txt', self.W)
					np.savetxt(output_directory_root+'W_out.txt', self.W_out)

				gc.collect()