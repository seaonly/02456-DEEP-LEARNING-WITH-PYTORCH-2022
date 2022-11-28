# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 12:43:22 2022

@author: nafan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 12:20:11 2022

@author: nafan
"""
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Conv2D,Add,Flatten,Dropout
from tensorflow.keras.layers import SeparableConv2D,ReLU
from tensorflow.keras.layers import BatchNormalization,MaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
#For fully connected networks
from tensorflow.keras.models import Sequential

#To construct the network we use dense connected layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Model
import numpy
import netCDF4 as nc
import xarray
import tensorflow_datasets as tfds
import os
import shutil
import glob
#import plotly.express as px
from tensorflow import keras
import pandas as pd
import csv
from tensorflow.keras.metrics import Precision,Recall,AUC,TruePositives,TrueNegatives,FalsePositives
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
dir_loc = 'N:/hpc/Data/'

def simple_load_nc_dir_with_generator(data, generate_tensor=True):
    #def gen():
        results_list = list()
        final_dict = None
        file_path = dir_loc + data +'.nc'
        #for file in glob.glob(os.path.join(dir_, "*.nc")):
        ds = xarray.open_dataset(file_path, engine='netcdf4')

        if generate_tensor == True:
            #final_dict = {'conv2d_input': results_list.append(tf.convert_to_tensor(val)) for key, val in ds.items()}
            #results_list = [tf.convert_to_tensor(val) for key, val in ds.items()]
            results_list += list(map( lambda x : tf.convert_to_tensor(x[1]) ,
                                     ds.items() ))
        else:
            #final_dict = {'conv2d_input': results_list + [val] for key, val in ds.items()}
            results_list += list(map( lambda x : x[1],
                                     ds.items() ))

        print("CCC", ds.items())

        #final_dict = { 'conv2d_input': results_list[0]  }
        results_list_shape = results_list[0].shape # (153, 41, 41)   
        final_dict = { # TODO: find function to add +1 dimension to tensor 
                    'conv2d_input': tf.reshape( results_list[0:5],
                                               shape=(results_list_shape[0],
                                                      results_list_shape[1], 
                                                      results_list_shape[2], 5)) }

        #assert(final_dict['input_1'].shape == (306, 281, 481, 2))

        #print("BBB", results_list[0] )
        print("X # of inputs:", final_dict['conv2d_input'].shape, "and", type(final_dict['conv2d_input']))

        return final_dict

X_values = simple_load_nc_dir_with_generator("x_train", generate_tensor=True)
print(type(X_values))
X_val = simple_load_nc_dir_with_generator("x_val", generate_tensor=True)
X_val['conv2d_input'].shape
X_test = simple_load_nc_dir_with_generator("x_test", generate_tensor=True)

def open_csv_file(data):
   
    Y_values = list()
    with open(dir_loc + data) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                #print(f'\t{row[0]} and {row[1]} and {row[2]}.')
                if row[1] == "Yes":
                    Y_values.append(1)
                else:
                    Y_values.append(0)

                line_count += 1
        print(f'Processed {line_count} lines.')

   
    Y_values = list(map(lambda x : tf.convert_to_tensor(x), Y_values))

    Y_values = {'dense_3': tf.convert_to_tensor( Y_values ) }


    return Y_values
Y_values = open_csv_file("y_train.csv") 
Y_val = open_csv_file("y_val.csv") 
Y_values['dense_3'].shape
Y_test = open_csv_file("y_test.csv") 

#The code was paerially taken from
# https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98
# Initialize the model object
model = Sequential()
# Add a convolutional layer
model.add(Conv2D(10, kernel_size=3, activation='relu', 
               input_shape=(281, 481, 5)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Flatten the output of the convolutional layer
model.add(Flatten())
# Add an output layer for the 3 categories
model.add(Dense(1, activation='sigmoid'))

model.summary()


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

model.compile(optimizer='adam',
          loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
          metrics=METRICS)

history = model.fit(
        X_values,
        Y_values,
        epochs=40,
        validation_data=(X_val, Y_val),
        batch_size=10)
model.evaluate(X_test, Y_test)



print(history.history.keys())
# summarize history for loss
plt.plot(history.history['prc'])
plt.plot(history.history['val_prc'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()