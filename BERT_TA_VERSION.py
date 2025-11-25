import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
import tensorflow_text
from transformers import CLIPTokenizer, TFCLIPTextModel
import tensorflow_hub as hub

import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import random
import time
from pathlib import Path
import re
from IPython import display

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

data_path = './dataset'
df = pd.read_pickle(data_path + '/text2ImgData.pkl')
num_training_sample = len(df)
n_images_train = num_training_sample
print('There are %d image in training data' % (n_images_train))

id2word_dict = dict(np.load('./dictionary/id2Word.npy'))

# Function to convert ids to words for one caption
def decode_caption(caption_id_list):
    return ' '.join([id2word_dict[str(id)] for id in caption_id_list if id2word_dict[str(id)] != '' and id2word_dict[str(id)] != '<PAD>' and id2word_dict[str(id)] != '<RARE>'])

# Each df['Captions'][i] is a list of 1-10 captions, each caption is a list of 20 word ids
# Decode all captions
all_captions_in_english = []
for caption_group in df['Captions']:
    english_captions = [decode_caption(caption) for caption in caption_group]
    all_captions_in_english.append(english_captions)

# Optionally: replace the captions in the dataframe
df['EnglishCaptions'] = all_captions_in_english

# in this competition, you have to generate image in size 64x64x3
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_CHANNEL = 3
IMAGE_SIZE_CROPPED = 48

def training_data_generator(caption, image_path):
    # load in the image according to image path
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, size=[IMAGE_HEIGHT, IMAGE_WIDTH])
    img.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])

    distorted_image = tf.image.random_crop(img, [IMAGE_SIZE_CROPPED,IMAGE_SIZE_CROPPED,IMAGE_CHANNEL])
    distorted_image = tf.image.resize(distorted_image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=0.2)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.8, upper=1.2)
    # distorted_image = tf.image.per_image_standardization(distorted_image)
    # distorted_image = tf.clip_by_value(distorted_image, 0.0, 1.0)
    ## newly added
    distorted_image = tf.clip_by_value(distorted_image, 0.0, 1.0)
    distorted_image = (distorted_image - 0.5) * 2.0  # scale to [-1, 1]
    return distorted_image, caption

def dataset_generator(filenames, batch_size, data_generator):
    # load the training data into two NumPy arrays
    df = pd.read_pickle(filenames)
    captions = df['Captions'].values
    caption = []
    sentence = []
    # each image has 1 to 10 corresponding captions
    # we choose one of them randomly for training
    for i in range(len(captions)):
        temp = random.choice(captions[i])
        caption.append(temp)
        s = decode_caption(temp)
        sentence.append(s)
        
    caption = np.asarray(sentence)
    image_path = df['ImagePath'].values
    
    # assume that each row of `features` corresponds to the same row as `labels`.
    assert caption.shape[0] == image_path.shape[0]
    
    dataset = tf.data.Dataset.from_tensor_slices((caption, image_path))
    dataset = dataset.map(data_generator, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(len(caption)).batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

BATCH_SIZE = 64
dataset = dataset_generator(data_path + '/text2ImgData.pkl', BATCH_SIZE, training_data_generator)

class ClipTextEncoder(tf.keras.Model):
    def __init__(self, hparas):
        super(ClipTextEncoder, self).__init__()
        model_name = "openai/clip-vit-base-patch32"
        
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.encoder = TFCLIPTextModel.from_pretrained(model_name)
        self.encoder.trainable = False

    def call(self, text):

        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            return_tensors="tf"
        )
        outputs = self.encoder(inputs)
        return outputs.pooler_output
    
class Generator(tf.keras.Model):
    """
    Generate fake image based on given text(hidden representation) and noise z
    input: text and noise
    output: fake image with size 64*64*3
    """
    def __init__(self, hparas):
        super(Generator, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        # strengthen hidden size
        self.d1 = tf.keras.layers.Dense(hparas['DENSE_DIM']*2)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.d2 = tf.keras.layers.Dense(hparas['DENSE_DIM'])
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.d3 = tf.keras.layers.Dense(hparas['DENSE_DIM'] // 2)
        self.d4 = tf.keras.layers.Dense(hparas['DENSE_DIM'] // 4)
        self.bn4 = tf.keras.layers.BatchNormalization()

        self.bn3 = tf.keras.layers.BatchNormalization()
        self.d_out = tf.keras.layers.Dense(64*64*3)
        
    def call(self, text, noise_z):
        x = self.flatten(text)
        x = self.d1(x)
        x = self.bn1(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = self.d2(x)
        x = self.bn2(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = self.d3(x)
        x = self.bn3(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = self.d4(x)
        x = self.bn4(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        # concatenate input text and random noise
        x = tf.concat([noise_z, x], axis=1)
        x = self.d_out(x)
        
        logits = tf.reshape(x, [-1, 64, 64, 3])
        output = tf.nn.tanh(logits)
        return logits, output

class Discriminator(tf.keras.Model):
    """
    Differentiate the real and fake image
    input: image and corresponding text
    output: labels, the real image should be 1, while the fake should be 0
    """
    def __init__(self, hparas):
        super(Discriminator, self).__init__()
        self.hparas = hparas
        self.flatten = tf.keras.layers.Flatten()
        self.d_text1 = tf.keras.layers.Dense(self.hparas['DENSE_DIM'])
        self.drop_text = tf.keras.layers.Dropout(0.3)
        self.bn_text1 = tf.keras.layers.BatchNormalization()
        self.d_text2 = tf.keras.layers.Dense(self.hparas['DENSE_DIM'])
        self.bn_text2 = tf.keras.layers.BatchNormalization()

        self.d_img1 = tf.keras.layers.Dense(self.hparas['DENSE_DIM'])
        self.drop_img = tf.keras.layers.Dropout(0.3)
        self.bn_img1 = tf.keras.layers.BatchNormalization()
        self.d_img2 = tf.keras.layers.Dense(self.hparas['DENSE_DIM'])
        self.bn_img2 = tf.keras.layers.BatchNormalization()

        self.d1 = tf.keras.layers.Dense(self.hparas['DENSE_DIM'])
        self.drop_joint = tf.keras.layers.Dropout(0.4)
        self.d2 = tf.keras.layers.Dense(1)

    def call(self, img, text, training=False):
        text = self.flatten(text)
        text = self.d_text1(text)
        text = tf.nn.leaky_relu(text, alpha=0.2)
        text = self.drop_text(text, training=training)

        img = self.flatten(img)
        img = self.d_img1(img)
        img = tf.nn.leaky_relu(img, alpha=0.2)
        img = self.drop_img(img, training=training)

        # concatenate image with paired text
        img_text = tf.concat([text, img], axis=1)
        img_text = self.d1(img_text)
        img_text = tf.nn.leaky_relu(img_text, alpha=0.2)
        img_text = self.drop_joint(img_text, training=training)

        logits = self.d2(img_text)
        output = tf.nn.sigmoid(logits)
        return logits, output

hparas = {
    'MAX_SEQ_LENGTH': 20,                     # maximum sequence length
    'Z_DIM': 512,                             # random noise z dimension
    'DENSE_DIM': 256,                         # number of neurons in dense layer
    'IMAGE_SIZE': [64, 64, 3],                # render image size
    'BATCH_SIZE': 64,
    'LR': 1e-4,
    'LR_DECAY': 0.5,
    'BETA_1': 0.5,
    'N_EPOCH': 5000,                            # number of epoch for demo
    'N_SAMPLE': num_training_sample,          # size of training data
}

text_encoder = ClipTextEncoder(hparas)
generator = Generator(hparas)
discriminator = Discriminator(hparas)

def encode_caption_py(caption_np):
    # caption_np: numpy array of bytes, shape (B,)
    cap_list = [c.decode("utf-8") for c in caption_np]
    emb = text_encoder(cap_list).numpy()   # convert to numpy array
    return emb

def encode_caption_tf(caption):
    emb = tf.numpy_function(
        encode_caption_py, 
        [caption], 
        tf.float32      # 回傳的是 embedding 向量
    )
    emb.set_shape((None, hparas['Z_DIM']))  # 讓 TF 知道 shape（很重要）
    return emb

dataset = dataset.map(
    lambda img, cap: (img, encode_caption_tf(cap)),
    num_parallel_calls=tf.data.AUTOTUNE
)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_logits, fake_logits):
    # output value of real image should be 1
    real_loss = cross_entropy(tf.ones_like(real_logits) * 0.9, real_logits)
    # output value of fake image should be 0
    fake_loss = cross_entropy(tf.zeros_like(fake_logits), fake_logits)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    # output value of fake image should be 0
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-3)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
epoch_var = tf.Variable(0, trainable=False, dtype=tf.int64, name="epoch")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 text_encoder=text_encoder,
                                 generator=generator,
                                 discriminator=discriminator,
                                 epoch_var=epoch_var)
manager = tf.train.CheckpointManager(checkpoint, './ckpts',  max_to_keep=20,
                                     checkpoint_name='GAN')

# newly added
steps_per_epoch = len(dataset) 
TOTAL_STEPS = int(steps_per_epoch * hparas['N_EPOCH'])
START_NOISE = 0.1        
print(f"(TOTAL_STEPS): {TOTAL_STEPS}")

@tf.function
def train_step(real_image, caption):
    # newly added
    current_step = tf.cast(discriminator_optimizer.iterations, tf.float32)
    decay_rate = tf.maximum(0.0, 1.0 - (current_step / TOTAL_STEPS))
    current_stddev = START_NOISE * decay_rate
    # newly added
    
    
    # Update D once # newly added (normalize noise)
    noise_d = tf.random.normal(shape=[hparas['BATCH_SIZE'], hparas['Z_DIM']])
    noise_d = tf.math.l2_normalize(noise_d, axis=1)
    with tf.GradientTape() as disc_tape:
        _, fake_image_d = generator(caption, noise_d, training=True)
        ## newly added
        img_noise = tf.random.normal(shape=tf.shape(real_image), mean=0.0, stddev=current_stddev)
        real_image = real_image + img_noise
        fake_image_d = fake_image_d + img_noise
        ## newly added
        real_logits, _ = discriminator(real_image, caption, training=True)
        fake_logits_d, _ = discriminator(fake_image_d, caption, training=True)
        d_loss = discriminator_loss(real_logits, fake_logits_d)

    grad_d = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(grad_d, discriminator.trainable_variables))

    # Update G twice
    g_loss = 0.0
    for _ in tf.range(2):
        noise_g = tf.random.normal(shape=[hparas['BATCH_SIZE'], hparas['Z_DIM']])
        # newly added (normalize noise)
        noise_g = tf.math.l2_normalize(noise_g, axis=1)
        with tf.GradientTape() as gen_tape:
            _, fake_image_g = generator(caption, noise_g, training=True)
            ## newly added
            img_noise_g = tf.random.normal(shape=tf.shape(fake_image_g), mean=0.0, stddev=current_stddev)
            fake_image_g = fake_image_g + img_noise_g
            ## newly added
            
            fake_logits_g, _ = discriminator(fake_image_g, caption, training=True)
            curr_g_loss = generator_loss(fake_logits_g)

        grad_g = gen_tape.gradient(curr_g_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(grad_g, generator.trainable_variables))
        g_loss += curr_g_loss

    g_loss = g_loss / 2.0 
    return g_loss, d_loss

@tf.function
def test_step(caption, noise):
    _, fake_image = generator(caption, noise)
    return fake_image

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def imsave(images, size, path):
    # getting the pixel values between [0, 1] to save it
    return plt.imsave(path, merge(images, size)*0.5 + 0.5)

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def sample_generator(caption, batch_size):
    caption = np.asarray(caption)
    dataset = tf.data.Dataset.from_tensor_slices(caption)
    dataset = dataset.batch(batch_size)
    return dataset

ni = int(np.ceil(np.sqrt(hparas['BATCH_SIZE'])))
sample_size = hparas['BATCH_SIZE']
sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, hparas['Z_DIM'])).astype(np.float32)
# newly added (normalize noise)
sample_seed = sample_seed / np.linalg.norm(sample_seed, axis=1, keepdims=True)
sample_sentence = ["the flower shown has yellow anther red pistil and bright red petals."] * int(sample_size/ni) + \
                  ["this flower has petals that are yellow, white and purple and has dark lines"] * int(sample_size/ni) + \
                  ["the petals on this flower are white with a yellow center"] * int(sample_size/ni) + \
                  ["this flower has a lot of small round pink petals."] * int(sample_size/ni) + \
                  ["this flower is orange in color, and has petals that are ruffled and rounded."] * int(sample_size/ni) + \
                  ["the flower has yellow petals and the center of it is brown."] * int(sample_size/ni) + \
                  ["this flower has petals that are blue and white."] * int(sample_size/ni) +\
                  ["these white flowers have petals that start off white in color and end in a white towards the tips."] * int(sample_size/ni)

sample_sentence = sample_generator(sample_sentence, hparas['BATCH_SIZE'])

sample_sentence = sample_sentence.map(
    lambda cap: encode_caption_tf(cap),
    num_parallel_calls=tf.data.AUTOTUNE
)

if not os.path.exists('samples/demo'):
    os.makedirs('samples/demo')

def train(dataset):
    # hidden state of RNN
    steps_per_epoch = int(hparas['N_SAMPLE']/hparas['BATCH_SIZE'])
    
    for epoch in range(hparas['N_EPOCH']):
        g_total_loss = 0
        d_total_loss = 0
        start = time.time()
        
        for image, caption in dataset:
            g_loss, d_loss = train_step(image, caption)
            g_total_loss += g_loss
            d_total_loss += d_loss
            
        time_tuple = time.localtime()
        time_string = time.strftime("%m/%d/%Y, %H:%M:%S", time_tuple)
            
        print("Epoch {}, gen_loss: {:.4f}, disc_loss: {:.4f}".format(epoch+1,
                                                                     g_total_loss/steps_per_epoch,
                                                                     d_total_loss/steps_per_epoch))
        print('Time for epoch {} is {:.4f} sec'.format(epoch+1, time.time()-start))
        
        epoch_var.assign_add(1)
        
        save_path = manager.save()
        print("Saved checkpoint for epoch {}: {}".format(int(epoch_var), save_path))
        
        # visualization
        for caption in sample_sentence:
            fake_image = test_step(caption, sample_seed)
        save_images(fake_image, [ni, ni], 'samples/demo/train_{:02d}.jpg'.format(epoch))

if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
    print(f"Restored from {manager.latest_checkpoint}")
else:
    print("Initializing from scratch")

train(dataset)

