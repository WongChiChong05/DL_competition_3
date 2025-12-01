# Created by Jimmy Yuan on 2019/11/28.
# Copyright © 2019 Jimmy Yuan. All rights reserved.
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
# ==============================================================================

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np
import scipy.spatial
import pickle

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_CHANNEL = 3
################# Please set BATCH_SIZE to 1, 2, 3, 7, 9, 21, 39 to avoid remainder #################
BATCH_SIZE = int(sys.argv[3])
################# Please set BATCH_SIZE to 1, 2, 3, 7, 9, 21, 39 to avoid remainder #################

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[3], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def read_dict(path):
    # read eval_metrics.pkl
    pkl_file = open(path, 'rb')
    repre_dict = pickle.load(pkl_file)
    pkl_file.close()
    return repre_dict

def create_data(inference_path):
    # return path of generated image
    infer_path = Path(inference_path).glob('*.jpg')
    infer_path = [str(path.resolve()) for path in infer_path]
    infer_path = np.asarray(infer_path)
    
    # return index of generated image
    index = [int(path.split('_')[-1].split('.')[0]) for path in infer_path]
    index = np.asarray(index)
    return infer_path, index
        
def testing_data_generator(image_path, index):
    # load in the image according to image path
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, size=[IMAGE_HEIGHT, IMAGE_WIDTH])
    img.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    return img, index

def testing_dataset_generator(img_path, index, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((img_path, index))
    dataset = dataset.map(testing_data_generator, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset        
        
def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG16(include_top=False, weights=None)
    
    output = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], output)
    return model

class VGG16(tf.keras.Model):
    def __init__(self, layers, trainable=False):
        super(VGG16, self).__init__()
        self.vgg =  vgg_layers(layers)
        self.vgg.trainable = trainable
        
        # trainable layers
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(4096, activation='relu')
        self.d2 = tf.keras.layers.Dense(2048, activation='relu')
        self.d3 = tf.keras.layers.Dense(102)
        
    # return the feature map of required layer
    def call(self, inputs):
        outputs = self.vgg(inputs)
        outputs = self.flatten(outputs)
        outputs = self.d1(outputs)
        outputs = self.d2(outputs)
        logits = self.d3(outputs)
        prediction = tf.nn.softmax(logits)
        return logits, prediction

@tf.function
def inference_step(images, model):
    repre, predictions = model(images)
    return repre, predictions

def inference(dataset, model):
    repre_dict = dict()
    pred = []
    for inference_images, idx_batch in dataset:
        inference_images = tf.keras.applications.vgg19.preprocess_input(inference_images * 255)
        repre_batch, predictions_batch = inference_step(inference_images, model)
        pred.append(predictions_batch)
        for repre, idx in zip(repre_batch.numpy(), idx_batch.numpy()):
            repre_dict[idx] = repre
    return repre_dict, pred

def similarity_score(repre_dict, infer_repre_dict):
    score = []
    for idx, repre in infer_repre_dict.items():
        s = scipy.spatial.distance.cosine(repre, repre_dict[idx])
        score.append(s)
    return score
    
def inception_score(p_yx, eps=1e-16):
    # pred is p(y|x)
    # calculate p(y)
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    # calculate KL divergence using log probabilities
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    # undo the log
    score = np.exp(avg_kl_d)
    return score

def main(args):
    # activation layer
    layers = ['block5_pool']
    vgg16 = VGG16(layers, trainable=False)
    
    # restore pre-trained model
    checkpoint_dir = './checkpoints_transfer_fine_tune'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(vgg16=vgg16)
    checkpoint.restore(checkpoint_dir + '/ckpt-5').expect_partial()
    
    # read eval_metrics.pkl
    repre_dict = read_dict('./eval_metrics.pkl')
    
    # create inference dataset
    inference_path = args[1]
    infer_path, index = create_data(inference_path)
    inference_ds = testing_dataset_generator(infer_path, index, BATCH_SIZE)
    
    # inference
    infer_repre_dict, infer_pred = inference(dataset=inference_ds, model=vgg16)
    infer_pred = np.asarray(infer_pred)
    infer_pred = np.reshape(infer_pred, (-1, infer_pred.shape[-1]))
    
    # calculate score
    # final score is based on cosine similarity and inception score
    sim_score = similarity_score(repre_dict, infer_repre_dict)
    i_score = inception_score(infer_pred)  
    score = sim_score + 0.5 * (1 / i_score)
    
    # write output file
    idx = [i for i in range(len(score))]
    result = pd.DataFrame({'score':score, 'id':idx})
    output_name = args[2]
    result.to_csv(output_name, index=False)
    print('--------------Evaluation Success-----------------')

main(sys.argv)