import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
import tensorflow_text
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

data_path = './dataset'
df = pd.read_pickle(data_path + '/text2ImgData.pkl')
num_training_sample = len(df)
n_images_train = num_training_sample
print('There are %d image in training data' % (n_images_train))

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
    distorted_image = tf.image.per_image_standardization(distorted_image)
    distorted_image = tf.clip_by_value(distorted_image, -1.0, 1.0)


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

import tensorflow_hub as hub

class BertTextEncoder(tf.keras.Model):
    def __init__(self, hparas):
        super(BertTextEncoder, self).__init__()
        self.hparas = hparas
        # 加载 BERT 预处理器和编码器
        self.preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        self.encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",
        trainable=False  # 非训练模式
        )
        self.proj = tf.keras.layers.Dense(hparas['DENSE_DIM'])
        
    def call(self, text):
        # text: list of strings
        text_inputs = self.preprocessor(text)
        outputs = self.encoder(text_inputs)
        pooled = outputs['pooled_output'] 
        # 只返回 [CLS] embedding
        return self.proj(pooled)

    # Bert 不需要式初始化 hidden state

class Generator(tf.keras.Model):
    """
    Generate fake image based on given text(hidden representation) and noise z
    input: text (batch, T, D) or (batch, D), noise_z (batch, Z_DIM)
    output: fake image with size 64x64x3 in [-1, 1]
    """
    def __init__(self, hparas):
        super(Generator, self).__init__()
        self.hparas = hparas
        self.flatten = tf.keras.layers.Flatten()

        # 把文字嵌入壓到固定維度，方便和 noise concat
        self.text_fc = tf.keras.layers.Dense(
            hparas['DENSE_DIM'],
            activation=None
        )

        # 將 [noise_z, text_vec] 映成小 feature map（4x4x256）
        self.fc = tf.keras.layers.Dense(
            4 * 4 * 256,
            use_bias=False
        )
        self.bn0 = tf.keras.layers.BatchNormalization()

        # 反卷積 / 上採樣堆疊：4x4 → 8x8 → 16x16 → 32x32
        self.deconv1 = tf.keras.layers.Conv2DTranspose(
            128, kernel_size=4, strides=2, padding='same', use_bias=False
        )
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.deconv2 = tf.keras.layers.Conv2DTranspose(
            64, kernel_size=4, strides=2, padding='same', use_bias=False
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.deconv3 = tf.keras.layers.Conv2DTranspose(
            32, kernel_size=4, strides=2, padding='same', use_bias=False
        )
        self.bn3 = tf.keras.layers.BatchNormalization()

        # 最後一層改成 UpSampling2D + Conv2D，減少棋盤格
        self.upsample4 = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')  # 32x32 → 64x64
        self.conv4 = tf.keras.layers.Conv2D(
            3, kernel_size=3, strides=1, padding='same', use_bias=True
        )

    def call(self, text, noise_z, training=False):
        # text 可能是 (B, T, D)（RNN output）或 (B, D)（BERT pooled）
        x = self.flatten(text)          # (B, ?)
        x = self.text_fc(x)             # (B, DENSE_DIM)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        # concat noise
        z = tf.concat([noise_z, x], axis=1)   # (B, Z_DIM + DENSE_DIM)

        # 映成 4x4x256
        x = self.fc(z)
        x = self.bn0(x, training=training)
        x = tf.nn.relu(x)
        x = tf.reshape(x, [-1, 4, 4, 256])

        # 4x4 → 8x8
        x = self.deconv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        # 8x8 → 16x16
        x = self.deconv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        # 16x16 → 32x32
        x = self.deconv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)

        # 32x32 → 64x64x3，用上採樣 + 普通卷積，減少 checkerboard
        x = self.upsample4(x)          # 64x64
        logits = self.conv4(x)         # (B, 64, 64, 3)
        output = tf.nn.tanh(logits)    # [-1, 1]

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

        # text branch
        self.d_text1 = tf.keras.layers.Dense(self.hparas['DENSE_DIM'])
        self.drop_text = tf.keras.layers.Dropout(0.2)
        # BN 可以留一層就好，避免太多 BN 讓訓練更不穩
        self.bn_text1 = tf.keras.layers.BatchNormalization()

        self.d_text2 = tf.keras.layers.Dense(self.hparas['DENSE_DIM'])

        # image branch
        self.d_img1 = tf.keras.layers.Dense(self.hparas['DENSE_DIM'])
        self.drop_img = tf.keras.layers.Dropout(0.2)
        self.bn_img1 = tf.keras.layers.BatchNormalization()

        self.d_img2 = tf.keras.layers.Dense(self.hparas['DENSE_DIM'])

        # joint branch
        self.d1 = tf.keras.layers.Dense(self.hparas['DENSE_DIM'])
        self.drop_joint = tf.keras.layers.Dropout(0.3)
        self.d2 = tf.keras.layers.Dense(1)

    def call(self, img, text, training=False):
        # text branch
        text = self.flatten(text)
        text = self.d_text1(text)
        text = self.bn_text1(text, training=training)
        text = tf.nn.leaky_relu(text, alpha=0.2)
        text = self.drop_text(text, training=training)

        text = self.d_text2(text)
        text = tf.nn.leaky_relu(text, alpha=0.2)
        # 第二層就不要再 BN + Dropout 了，讓訊號乾淨一點

        # image branch
        img = self.flatten(img)
        img = self.d_img1(img)
        img = self.bn_img1(img, training=training)
        img = tf.nn.leaky_relu(img, alpha=0.2)
        img = self.drop_img(img, training=training)

        img = self.d_img2(img)
        img = tf.nn.leaky_relu(img, alpha=0.2)

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
    'PRINT_FREQ': 1                           # printing frequency of loss
}

text_encoder = BertTextEncoder(hparas)
generator = Generator(hparas)
discriminator = Discriminator(hparas)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_logits, fake_logits):
    # output value of real image should be 1
    real_loss = cross_entropy(tf.ones_like(real_logits) * 0.9, real_logits)
    # output value of fake image should be 0
    fake_loss = cross_entropy(tf.zeros_like(fake_logits), fake_logits)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_logits):
    return cross_entropy(tf.ones_like(fake_logits), fake_logits)

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
epoch_var = tf.Variable(0, trainable=False, dtype=tf.int64, name="epoch")

# one benefit of tf.train.Checkpoint() API is we can save everything seperately
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 text_encoder=text_encoder,
                                 generator=generator,
                                 discriminator=discriminator,
                                 epoch_var=epoch_var)
manager = tf.train.CheckpointManager(checkpoint, './ckpts',  max_to_keep=20,
                                     checkpoint_name='GAN')

@tf.function
def train_step(real_image, caption):
    text_embed = text_encoder(caption)

    # === 先更新 D 一次 ===
    noise_d = tf.random.normal(
        shape=[hparas['BATCH_SIZE'], hparas['Z_DIM']],
        mean=0.0, stddev=1.0
    )
    with tf.GradientTape() as disc_tape:
        _, fake_image_d = generator(text_embed, noise_d, training=True)
        real_logits, _ = discriminator(real_image, text_embed, training=True)
        fake_logits_d, _ = discriminator(fake_image_d, text_embed, training=True)
        d_loss = discriminator_loss(real_logits, fake_logits_d)

    grad_d = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(grad_d, discriminator.trainable_variables))

    # === 再更新 G 2次 ===
    g_loss = 0.0
    for _ in tf.range(2):
        noise_g = tf.random.normal(
            shape=[hparas['BATCH_SIZE'], hparas['Z_DIM']],
            mean=0.0, stddev=1.0
        )
        with tf.GradientTape() as gen_tape:
            _, fake_image_g = generator(text_embed, noise_g, training=True)
            fake_logits_g, _ = discriminator(fake_image_g, text_embed, training=True)
            curr_g_loss = generator_loss(fake_logits_g)

        grad_g = gen_tape.gradient(curr_g_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(grad_g, generator.trainable_variables))
        g_loss += curr_g_loss

    g_loss = g_loss / 2.0  # 回傳平均的 G loss
    return g_loss, d_loss

@tf.function
def test_step(caption, noise):
    text_embed = text_encoder(caption)
    _, fake_image = generator(text_embed, noise)
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
sample_sentence = ["the flower shown has yellow anther red pistil and bright red petals."] * int(sample_size/ni) + \
                  ["this flower has petals that are yellow, white and purple and has dark lines"] * int(sample_size/ni) + \
                  ["the petals on this flower are white with a yellow center"] * int(sample_size/ni) + \
                  ["this flower has a lot of small round pink petals."] * int(sample_size/ni) + \
                  ["this flower is orange in color, and has petals that are ruffled and rounded."] * int(sample_size/ni) + \
                  ["the flower has yellow petals and the center of it is brown."] * int(sample_size/ni) + \
                  ["this flower has petals that are blue and white."] * int(sample_size/ni) +\
                  ["these white flowers have petals that start off white in color and end in a white towards the tips."] * int(sample_size/ni)

sample_sentence = sample_generator(sample_sentence, hparas['BATCH_SIZE'])

if not os.path.exists('samples/demo'):
    os.makedirs('samples/demo')

def train(dataset):
    # hidden state of RNN
    steps_per_epoch = int(hparas['N_SAMPLE']/hparas['BATCH_SIZE'])
    
    start_epoch = int(epoch_var.numpy())
    print(f"Start training from epoch {start_epoch}")

    for epoch in range(start_epoch, hparas['N_EPOCH']):
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
        if (epoch + 1) % hparas['PRINT_FREQ'] == 0:
            for caption in sample_sentence:
                fake_image = test_step(caption, sample_seed)
            save_images(fake_image, [ni, ni], 'samples/demo/train_{:02d}.jpg'.format(epoch))

if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
    print(f"Restored from {manager.latest_checkpoint}")
else:
    print("Initializing from scratch")

train(dataset)


def testing_data_generator(caption, index):
    caption = tf.cast(caption, tf.float32)
    return caption, index

def testing_dataset_generator(batch_size, data_generator):
    data = pd.read_pickle('./dataset/testData.pkl')
    captions = data['Captions'].values
    caption = []
    sentence = []
    for i in range(len(captions)):
        temp = captions[i]
        caption.append(temp)
        s = decode_caption(temp)
        sentence.append(s)
    captions = np.asarray(sentence)
    index = np.asarray(data['ID'].values)

    dataset = tf.data.Dataset.from_tensor_slices((captions, index))
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset
    
testing_dataset = testing_dataset_generator(hparas['BATCH_SIZE'], testing_data_generator)

data = pd.read_pickle('./dataset/testData.pkl')
captions = data['Captions'].values

NUM_TEST = len(captions)
EPOCH_TEST = int(NUM_TEST / hparas['BATCH_SIZE'])

def inference(dataset):
    sample_size = hparas['BATCH_SIZE']
    # 這裡生成的 sample_seed 大小是 (64, 512)
    sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, hparas['Z_DIM'])).astype(np.float32)
    
    step = 0
    start = time.time()
    for captions, idx in dataset:
        if step > EPOCH_TEST:
            break
        
        # 1. 取得當前 batch 的實際大小
        current_batch_size = captions.shape[0]
        
        # 2. 根據當前 batch size 切割 sample_seed (或生成新的 noise)
        # 這樣即使最後一個 batch 只有 51 筆，也能正確對應
        current_noise = sample_seed[:current_batch_size]
        
        # 3. 使用調整後大小的 noise 傳入 test_step
        fake_image = test_step(captions, current_noise)
        
        step += 1
        
        # 4. 存檔時也要用 current_batch_size 避免 index error
        for i in range(current_batch_size):
            plt.imsave('./inference/demo/inference_{:04d}.jpg'.format(idx[i]), fake_image[i].numpy()*0.5 + 0.5)
            
    print('Time for inference is {:.4f} sec'.format(time.time()-start))

