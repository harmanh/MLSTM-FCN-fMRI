import os
import sys
import code
import random
import re
import pickle

import numpy as np
np.warnings.filterwarnings('ignore')
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras import optimizers
from keras.models import Model, Sequential, load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, TensorBoard
#code.interact(local=locals())

# Squeeze and excite block
def SqE_block(x, ratio=16):
    shape = x.shape.as_list()
    filters = shape[-1]
    z = GlobalAveragePooling1D()(x)
    s = Dense(filters // ratio, activation='relu', use_bias=False)(z)
    s = Dense(filters, activation='sigmoid', use_bias=False)(s)
    x = Multiply()([x, s])
    return x

# https://stackoverflow.com/a/34325723
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

# If the data hasn't been saved import ./data/pheno/pheno.csv & ./data/fmri/*.csv
if not os.path.isfile('input.npy'):
	pheno_file	= os.path.join(os.getcwd(), r'data', r'pheno', r'pheno.csv')
	pheno_data	= np.genfromtxt(pheno_file, dtype='i4', skip_header = 1, usecols=(2,7), delimiter=",")

	# Load list of [ID, csv_filename]
	IDs = []

	fmri_dir = os.path.join(os.getcwd(), r'data', r'fmri')

	for file in os.listdir(fmri_dir):
		if os.path.splitext(file)[-1].lower() == '.csv':
			m = re.search(r"(?<!\d)\d{7}(?!\d)", file)
			n = int(m.group(0))
			IDs.append([n, os.path.join(fmri_dir, file)])
			
	# Load list of [ndarray, ASD(Y=1 N=0)] to be split into test/train
	fmri_data = []
	max_tstep = 0
	max_regns = 0

	printProgressBar(0, len(IDs), prefix = 'Loading data:', suffix = 'Complete', length = 50)
	for i, id in enumerate(IDs):
		ASD = -1*pheno_data[np.where(pheno_data[:,0] == id[0])][0][1]+2		# 2 -> 0; 1 -> 1
		
		dataframe 	= pd.read_csv(id[1])
		dataset		= dataframe.values
		dataset 	= dataset.astype('float32')
		dataset		= normalize(dataset)
		
		fmri_data.append([dataset, ASD])
		
		if dataset.shape[0] > max_tstep:
			max_tstep = dataset.shape[0]
			
		if dataset.shape[1] > max_regns:
			max_regns = dataset.shape[1]
		
		printProgressBar(i+1, len(IDs), prefix = 'Loading data:', suffix = 'Complete', length = 50)	

	fmri_data = np.array(fmri_data)

	# Pad data
	for i, seq in enumerate(fmri_data[:,0]):
		fmri_data[:,0][i] = np.pad(seq, ((0, max_tstep-seq.shape[0]), (0, max_regns-seq.shape[1])), mode='constant', constant_values=0.0)

	np.save('input', fmri_data)
else:
	fmri_data = np.load('input.npy')	
	max_tstep = fmri_data[0][0].shape[0]
	max_regns = fmri_data[0][0].shape[1]

# Input tensor of [batches, tstep, variate] output binary classification
X = np.rollaxis(np.dstack(fmri_data[:,0]),-1)
y = fmri_data[:,1]
'''
# Test = USM & Yale
X_train = X[:907,:,:]
X_test  = X[908:,:,:]
y_train = y[:907]
y_test  = y[908:]
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=3)

if not os.path.isfile('model-best.hdf5'):
	inputs = Input(shape=X.shape[1:])
	
	lstm = CuDNNLSTM(256, return_sequences=True)(inputs)
	lstm = CuDNNLSTM(256, return_sequences=True)(lstm)
	lstm = CuDNNLSTM(256, return_sequences=True)(lstm)
	lstm = CuDNNLSTM(256, return_sequences=True)(lstm)
	lstm = CuDNNLSTM(256, return_sequences=True)(lstm)
	lstm = CuDNNLSTM(256, return_sequences=True)(lstm)
	lstm = CuDNNLSTM(256, return_sequences=True)(lstm)
	lstm = CuDNNLSTM(256)(lstm)
	lstm = Dropout(0.6)(lstm)
	
	conv = Conv1D(32, 5, padding='causal', kernel_initializer='he_uniform')(inputs)
	conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	conv = SqE_block(conv)
	conv = Conv1D(64, 5, padding='causal', kernel_initializer='he_uniform')(conv)
	conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	conv = SqE_block(conv)
	conv = Conv1D(128, 5, padding='causal', kernel_initializer='he_uniform')(conv)
	conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)
	conv = SqE_block(conv)
	conv = Conv1D(256, 5, padding='causal', kernel_initializer='he_uniform')(conv)
	conv = BatchNormalization()(conv)
	conv = Activation('relu')(conv)	
	conv = GlobalAveragePooling1D()(conv)

	concat = concatenate([conv, lstm])

	output = Dense(256, activation='relu')(concat)
	output = Dense(256, activation='relu')(output)
	output = Dense(1, activation='sigmoid')(concat)

	model = Model(inputs=inputs, outputs=output)
	resumed = False
else:
	print("\nResuming training from saved model\n")
	model = load_model('model-best.hdf5')
	resumed = True

print(model.summary(), "\n\nPress enter to fit model", end="")
input()

# Run model; register callbacks, compile, then fit

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

bsize = 64

tbCallBack = TensorBoard(log_dir='D:\TBoard', batch_size=bsize, histogram_freq=8, write_images=True)
checkBest = ModelCheckpoint('model-best.hdf5', monitor='val_acc', save_best_only=True)
checkLast = ModelCheckpoint('model-last.hdf5', period=10)

nosave = True
if not nosave:
	cbList = [checkBest, checkLast, tbCallBack]
else:
	cbList = []

hist = model.fit(X_train, y_train, nb_epoch=250, batch_size=bsize, validation_data=[X_test, y_test], callbacks=cbList)

if os.path.isfile('trainHistory.dict') and resumed:
	with open('trainHistory.dict', 'wb+') as file:
		old_history = pickle.load(file)
		old_history.update(hist.history)
		pickle.dump(old_history, file)
else:
	with open('trainHistory.dict', 'wb') as file:
		pickle.dump(hist.history, file)

