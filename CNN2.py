# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 07:22:01 2022

@author: nafan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:02:00 2022

@author: nafan
"""

import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Conv2D,Add
from tensorflow.keras.layers import SeparableConv2D,ReLU
from tensorflow.keras.layers import BatchNormalization,MaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
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
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import AUC
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

        results_list_shape = results_list[0].shape   
        final_dict = { 'input_1': tf.reshape( results_list[0:5],
                                               shape=(results_list_shape[0],
                                                      results_list_shape[1], 
                                                      results_list_shape[2], 5)) }

        print("X # of inputs:", final_dict['input_1'].shape, "and", type(final_dict['input_1']))

        return final_dict

X_values = simple_load_nc_dir_with_generator("x_train", generate_tensor=True)
print(type(X_values))
X_val = simple_load_nc_dir_with_generator("x_val", generate_tensor=True)
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

    Y_values = {'dense': tf.convert_to_tensor( Y_values ) }


    return Y_values
Y_values = open_csv_file("y_train.csv") 
Y_val = open_csv_file("y_val.csv") 
Y_test = open_csv_file("y_test.csv") 

# creating the Conv-Batch Norm block
#This part is partially taken 
#from https://towardsdatascience.com/xception-from-scratch-using-tensorflow-even-better-than-inception-940fb231ced9

def conv_bn(x, filters, kernel_size, strides=1):
    
    x = Conv2D(filters=filters, 
               kernel_size = kernel_size, 
               strides=strides, 
               padding = 'same', 
               use_bias = False)(x)
    x = BatchNormalization()(x)
    return x
# creating separableConv-Batch Norm block

def sep_bn(x, filters, kernel_size, strides=1):
    
    x = SeparableConv2D(filters=filters, 
                        kernel_size = kernel_size, 
                        strides=strides, 
                        padding = 'same', 
                        use_bias = False)(x)
    x = BatchNormalization()(x)
    return x
# entry flow

def entry_flow(x):
    
    x = conv_bn(x, filters =32, kernel_size =3, strides=2)
    x = ReLU()(x)
    x = conv_bn(x, filters =64, kernel_size =3, strides=1)
    tensor = ReLU()(x)
    
    x = sep_bn(tensor, filters = 128, kernel_size =3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 128, kernel_size =3)
    x = MaxPool2D(pool_size=3, strides=2, padding = 'same')(x)
    
    tensor = conv_bn(tensor, filters=128, kernel_size = 1,strides=2)
    x = Add()([tensor,x])
    
    x = ReLU()(x)
    x = sep_bn(x, filters =256, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters =256, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding = 'same')(x)
    
    tensor = conv_bn(tensor, filters=256, kernel_size = 1,strides=2)
    x = Add()([tensor,x])
    
    x = ReLU()(x)
    x = sep_bn(x, filters =728, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters =728, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding = 'same')(x)
    
    tensor = conv_bn(tensor, filters=728, kernel_size = 1,strides=2)
    x = Add()([tensor,x])
    return x
# middle flow

def middle_flow(tensor):
    
    for _ in range(8):
        x = ReLU()(tensor)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        x = sep_bn(x, filters = 728, kernel_size = 3)
        x = ReLU()(x)
        tensor = Add()([tensor,x])
        
    return tensor
# exit flow

def exit_flow(tensor):
    
    x = ReLU()(tensor)
    x = sep_bn(x, filters = 728,  kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 1024,  kernel_size=3)
    x = MaxPool2D(pool_size = 3, strides = 2, padding ='same')(x)
    
    tensor = conv_bn(tensor, filters =1024, kernel_size=1, strides =2)
    x = Add()([tensor,x])
    
    x = sep_bn(x, filters = 1536,  kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters = 2048,  kernel_size=3)
    x = GlobalAvgPool2D()(x)
    
    x = Dense (units = 1, activation = 'sigmoid')(x)
    
    return x
# model code

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

# Run training on CPU, because the GPU runs out of memory quickly and crashes the program
with tf.device('/cpu:0'):

    input = Input(shape = (281, 481, 5))
    x = entry_flow(input)
    x = middle_flow(x)
    output = exit_flow(x)

    model = Model(inputs=input, outputs=output)
    model.summary()

    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=METRICS)
    
    history = model.fit(
            X_values,
            Y_values,
            epochs=40,
            validation_data=(X_val, Y_val),
            batch_size=10
            )
    # Fit the model
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









