# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example implementation of code to run on the Cloud ML service.
"""

import argparse
import logging
import json
import os
import tensorflow as tf
from . import model

if __name__ == '__main__':
  print("Using TensorFlow version: {}".format(tf.version.VERSION))
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--project',
      help='the project id',
      required=True
  )
  parser.add_argument(
      '--bucket',
      help='Training data will be in gs://experiment_v1/TrainingData',
      required=True
  )
  parser.add_argument(
      '--job-dir',
      help='this model ignores this field, but it is required by gcloud',
      default='./junk'
  )
  
  parser.add_argument(
      '--num_examples',
      help='Number of examples to train on; this could be size-of-data X number-epochs',
      default=100
  )

  
  parser.add_argument(
      '--func',
      help='read_lines find_average_label linear OR wide_deep (default)',
      default='train_and_evaluate'
  )
  
  # parse args
  args = parser.parse_args()
  arguments = args.__dict__

  # unused args provided by service
  arguments.pop('job-dir', None)
  arguments.pop('job_dir', None)
    
  # set appropriate output directory
  PROJECT =  arguments['project']
  BUCKET = arguments['bucket']
  output_dir='gs://{}/Model/trained_model'.format(BUCKET)
  
  
  print('Writing trained model to {}'.format(output_dir))
  arguments['output_dir'] = output_dir
 

  # run
  logging.basicConfig(level=logging.INFO)
  model.setup(arguments)

  func_to_call = arguments['func']

  if func_to_call == 'read_lines':
    model.read_lines()
  elif func_to_call == 'find_average_label':
    model.find_average_label()
  else:  # linear, wide_deep
    model.train_and_evaluate()
                          
