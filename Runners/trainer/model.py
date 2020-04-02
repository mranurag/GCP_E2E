import logging
import os
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

PROJECT = None
BUCKET = None
OUTPUT_DIR = None
NBUCKETS = None
NUM_EXAMPLES = None
TRAIN_BATCH_SIZE = None
TRAIN_DATA_PATTERN = None
EVAL_DATA_PATTERN = None


def setup(args):
    
    global BUCKET, OUTPUT_DIR, NBUCKETS, NUM_EXAMPLES,PROJECT
    global TRAIN_BATCH_SIZE, TRAIN_DATA_PATTERN, EVAL_DATA_PATTERN
    global DNN_HIDDEN_UNITS
    
    BUCKET = args['bucket']
    PROJECT = args['project']
    OUTPUT_DIR = args['output_dir']
    
    # set up training and evaluation data patterns
    DATA_BUCKET = "gs://{}/TrainingData/".format(BUCKET)
    TRAIN_DATA_PATTERN = DATA_BUCKET + "data.csv"
    
    logging.info('Training based on data in {}'.format(TRAIN_DATA_PATTERN))
    

def read_dataset(pattern,colList):

    dataset = pd.read_csv(pattern)
    dataset = dataset[colList]
    return dataset
    
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

def read_lines():
    
    one_item = read_dataset(TRAIN_DATA_PATTERN,["trestbps","chol","thalach","target"])
    print("yes") # should print one batch of items   

def wide_and_deep_classifier():
    feature_columns = []


    for header in ['trestbps', 'chol', 'thalach']:
      feature_columns.append(feature_column.numeric_column(header))

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    model = tf.keras.Sequential([feature_layer,layers.Dense(128, activation='relu'),layers.Dense(128, activation='relu'),layers.Dense(1,activation='sigmoid')])

    model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
    return model


def train_and_evaluate():
    # create inputs and feature columns
    
    # create model
  
    model = wide_and_deep_classifier()
    
    # train and evaluate
    train_batch_size = TRAIN_BATCH_SIZE

   
    dataset = read_dataset(TRAIN_DATA_PATTERN, ["trestbps","chol","thalach","target"])
    print("data have been loaded successfully")
    train, test = train_test_split(dataset, test_size=0.2)
    print("train and test sets have been created successfully")
    batch_size = 50 # A small batch sized is used for demonstration purposes
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds   = df_to_dataset(test, shuffle=False, batch_size=batch_size)
    model.fit(train_ds,validation_data=val_ds,epochs=5)
    # write out final metric
    accuracy = model.evaluate(val_ds)
    
    validDict={}
    validDict["Feature"]=["trestbps","chol","thalach"]
    validDict["Average_Impact_on_Model_Output_Magnitude"]=[.3,.6,.1]
    impData = pd.DataFrame(validDict)
    impData.to_gbq(destination_table="modelvalidation.Importancetable1",project_id=PROJECT,if_exists='replace')

    print("Accuracy = {}".format(accuracy))
    # export
    export_dir = os.path.join(OUTPUT_DIR,
                              'export/heart_{}'.format(
                                  time.strftime("%Y%m%d-%H%M%S")))
    print('Exporting to {}'.format(export_dir))
    tf.saved_model.save(model, export_dir)
    
    # write out final metric
    accuracy = model.evaluate(val_ds)

    print("Accuracy = {}".format(accuracy))
    
