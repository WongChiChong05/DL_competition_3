
import os
import warnings
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import CLIPTokenizer, TFCLIPTextModel

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
import math


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

# Feel free t
dataset_repetitions = 1
num_epochs = 5000 
image_size = 64
suffle_times = 10

# KID = Kernel Inception Distance, see related section
kid_image_size = 75
kid_diffusion_steps = 5
plot_diffusion_steps = 20

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 2

# optimization
batch_size = 32
ema = 0.999
learning_rate = 1e-4
weight_decay = 1e-4

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

i = 0
for group in df['EnglishCaptions']:
    i += 1
    print(f'Image {i} captions:')
    for caption in group:
        print(caption)

# in this competition, you have to generate image in size 64x64x3
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_CHANNEL = 3
IMAGE_SIZE_CROPPED = 48
model_name = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(model_name)

def training_data_generator(input_ids, image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32) 
    
    # Resize & Crop
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, size=[64, 64])
    img.set_shape([64, 64, 3])
    
    # Data Augmentation
    # distorted_image = tf.image.random_crop(img, [48, 48, 3])
    # distorted_image = tf.image.resize(distorted_image, [64, 64])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=0.2)
    distorted_image = tf.clip_by_value(distorted_image, 0.0, 1.0)
    
    return input_ids, distorted_image

def dataset_generator(filenames, batch_size):
    df = pd.read_pickle(filenames)
    captions = df['Captions'].values
    
    sentences = []
    for i in range(len(captions)):
        temp = random.choice(captions[i])
        sentences.append(decode_caption(temp))
        
    print("Tokenizing all sentences...")
    encodings = tokenizer(
        sentences, padding="max_length", truncation=True, max_length=77, return_tensors="np"
    )
    all_input_ids = encodings['input_ids']
    image_paths = df['ImagePath'].values

    dataset = tf.data.Dataset.from_tensor_slices((all_input_ids, image_paths))
    dataset = dataset.map(training_data_generator, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.shuffle(len(sentences)).batch(batch_size, drop_remainder=True).repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

BATCH_SIZE = 64
dataset = dataset_generator(data_path + '/text2ImgData.pkl', BATCH_SIZE)

class ClipTextEncoder(tf.keras.Model):
    def __init__(self):
        super(ClipTextEncoder, self).__init__()
        self.encoder = TFCLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.encoder.trainable = False

    def call(self, input_ids):
        return self.encoder(input_ids).pooler_output

@keras.utils.register_keras_serializable()
def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.math.exp(
        tf.linspace(tf.math.log(embedding_min_frequency), tf.math.log(embedding_max_frequency), embedding_dims // 2)
    )
    angular_speeds = tf.cast(2.0 * math.pi * frequencies, "float32")
    embeddings = tf.concat([tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3)
    return embeddings

# ==========================================
# 修正後的模型定義：加入 build 方法確保權重追蹤
# ==========================================

# 1. ResidualBlock (大幅修正)
# ==========================================
# 1. 修改 ResidualBlock：加入 text_emb 參數
# ==========================================
class ResidualBlock(layers.Layer):
    def __init__(self, width, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        
        self.conv_1 = layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")
        self.conv_2 = layers.Conv2D(width, kernel_size=3, padding="same")
        self.conv_skip = layers.Conv2D(width, kernel_size=1)
        self.bn = layers.BatchNormalization(center=False, scale=False)
        self.add_layer = layers.Add()
        
        # [新增] 用來投影文字 Embedding 的全連接層
        # 我們要把文字向量 (例如 512維) 轉成跟圖片一樣的 Channel 數 (例如 width)
        self.text_proj = layers.Dense(width, activation="swish")

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        # inputs 包含 [x, text_emb]
        x, text_emb = inputs 
        
        input_width = x.shape[-1]
        
        if input_width == self.width:
            residual = x
        else:
            residual = self.conv_skip(x)

        y = self.bn(x)
        y = self.conv_1(y)
        
        # [關鍵修改] 注入文字特徵
        if text_emb is not None:
            # 1. 投影文字向量: (Batch, TextDim) -> (Batch, Width)
            t = self.text_proj(text_emb) 
            # 2. 改變形狀以便與圖片相加: (Batch, 1, 1, Width)
            t = tf.reshape(t, [-1, 1, 1, self.width])
            # 3. 加到圖片特徵上
            y = layers.Add()([y, t])

        y = self.conv_2(y)
        return self.add_layer([y, residual])

# ==========================================
# 2. 修改 DownBlock：傳遞 text_emb
# ==========================================
def down_block(x, text_emb, skips, width, block_depth):
    for _ in range(block_depth):
        # 修改：傳入 list [x, text_emb]
        x = ResidualBlock(width)([x, text_emb]) 
        skips.append(x)
    x = layers.AveragePooling2D(pool_size=2)(x)
    return x

# ==========================================
# 3. 修改 UpBlock：傳遞 text_emb
# ==========================================
def up_block(x, text_emb, skips, width, block_depth):
    x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
    for _ in range(block_depth):
        x = layers.Concatenate()([x, skips.pop()])
        # 修改：傳入 list [x, text_emb]
        x = ResidualBlock(width)([x, text_emb])
    return x

# ==========================================
# 4. 修改 get_network：配置文字注入路徑
# ==========================================
def get_network(image_size, widths, block_depth, text_embedding_dim=512):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))
    text_embeddings = keras.Input(shape=(text_embedding_dim,))

    # 時間 Embedding (保持不變)
    e = layers.Lambda(sinusoidal_embedding, output_shape=(1, 1, 32))(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    # 初始合併 (保留原本的合併，當作 Base)
    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e]) 
    # 注意：這裡我不把 text 放在 concatenate 了，因為我們要在後面 Deep Injection

    skips = []
    
    # Downpath
    for width in widths[:-1]:
        # 修改：把 text_embeddings 傳進去
        x = down_block(x, text_embeddings, skips, width, block_depth)

    # Bottleneck
    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])([x, text_embeddings])

    # Uppath
    for width in reversed(widths[:-1]):
        x = up_block(x, text_embeddings, skips, width, block_depth)

    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances, text_embeddings], x, name="residual_unet")

# ==========================================
# 1. 修改 ResidualBlock：加入 text_emb 參數
# ==========================================
class ResidualBlock(layers.Layer):
    def __init__(self, width, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        
        self.conv_1 = layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")
        self.conv_2 = layers.Conv2D(width, kernel_size=3, padding="same")
        self.conv_skip = layers.Conv2D(width, kernel_size=1)
        self.bn = layers.BatchNormalization(center=False, scale=False)
        self.add_layer = layers.Add()
        
        # [新增] 用來投影文字 Embedding 的全連接層
        # 我們要把文字向量 (例如 512維) 轉成跟圖片一樣的 Channel 數 (例如 width)
        self.text_proj = layers.Dense(width, activation="swish")

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        # inputs 包含 [x, text_emb]
        x, text_emb = inputs 
        
        input_width = x.shape[-1]
        
        if input_width == self.width:
            residual = x
        else:
            residual = self.conv_skip(x)

        y = self.bn(x)
        y = self.conv_1(y)
        
        # [關鍵修改] 注入文字特徵
        if text_emb is not None:
            # 1. 投影文字向量: (Batch, TextDim) -> (Batch, Width)
            t = self.text_proj(text_emb) 
            # 2. 改變形狀以便與圖片相加: (Batch, 1, 1, Width)
            t = tf.reshape(t, [-1, 1, 1, self.width])
            # 3. 加到圖片特徵上
            y = layers.Add()([y, t])

        y = self.conv_2(y)
        return self.add_layer([y, residual])

# ==========================================
# 2. 修改 DownBlock：傳遞 text_emb
# ==========================================
def down_block(x, text_emb, skips, width, block_depth):
    for _ in range(block_depth):
        # 修改：傳入 list [x, text_emb]
        x = ResidualBlock(width)([x, text_emb]) 
        skips.append(x)
    x = layers.AveragePooling2D(pool_size=2)(x)
    return x

# ==========================================
# 3. 修改 UpBlock：傳遞 text_emb
# ==========================================
def up_block(x, text_emb, skips, width, block_depth):
    x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
    for _ in range(block_depth):
        x = layers.Concatenate()([x, skips.pop()])
        # 修改：傳入 list [x, text_emb]
        x = ResidualBlock(width)([x, text_emb])
    return x

# ==========================================
# 4. 修改 get_network：配置文字注入路徑
# ==========================================
def get_network(image_size, widths, block_depth, text_embedding_dim=512):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))
    text_embeddings = keras.Input(shape=(text_embedding_dim,))

    # 時間 Embedding (保持不變)
    e = layers.Lambda(sinusoidal_embedding, output_shape=(1, 1, 32))(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    # 初始合併 (保留原本的合併，當作 Base)
    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e]) 
    # 注意：這裡我不把 text 放在 concatenate 了，因為我們要在後面 Deep Injection

    skips = []
    
    # Downpath
    for width in widths[:-1]:
        # 修改：把 text_embeddings 傳進去
        x = down_block(x, text_embeddings, skips, width, block_depth)

    # Bottleneck
    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])([x, text_embeddings])

    # Uppath
    for width in reversed(widths[:-1]):
        x = up_block(x, text_embeddings, skips, width, block_depth)

    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances, text_embeddings], x, name="residual_unet")

sample_sentences = [
    "the flower shown has yellow anther red pistil and bright red petals.",
    "this flower has petals that are yellow, white and purple and has dark lines",
    "the petals on this flower are white with a yellow center",
    "this flower has a lot of small round pink petals.",
    "this flower is orange in color, and has petals that are ruffled and rounded.",
    "the flower has yellow petals and the center of it is brown.",
    "this flower has petals that are blue and white.",
    "these white flowers have petals that start off white in color and end in a white towards the tips."
]


print("Tokenizing sample sentences...")
encodings = tokenizer(
    sample_sentences, 
    padding="max_length", 
    truncation=True, 
    max_length=77,
    return_tensors="tf"
)
sample_input_ids = encodings['input_ids']

print("sample_input_ids ready!")

# ==========================================
# 必修修正：將模型區塊改為 Class，確保權重被訓練
# ==========================================

# Embedding 函數保持不變
@keras.utils.register_keras_serializable()
def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.math.exp(
        tf.linspace(tf.math.log(embedding_min_frequency), tf.math.log(embedding_max_frequency), embedding_dims // 2)
    )
    angular_speeds = tf.cast(2.0 * math.pi * frequencies, "float32")
    embeddings = tf.concat([tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3)
    return embeddings


class ResidualBlock(layers.Layer):
    def __init__(self, width, **kwargs):
        super().__init__(**kwargs)
        self.width = width
  
        self.conv_1 = layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")
        self.conv_2 = layers.Conv2D(width, kernel_size=3, padding="same")
        
       
        self.conv_skip = layers.Conv2D(width, kernel_size=1)
        
        self.bn = layers.BatchNormalization(center=False, scale=False)
        self.add_layer = layers.Add() 

    def build(self, input_shape):
        
        super().build(input_shape)

    def call(self, x):
        input_width = x.shape[-1] 
        
        if input_width == self.width:
            residual = x
        else:
            residual = self.conv_skip(x)

        y = self.bn(x)
        y = self.conv_1(y)
        y = self.conv_2(y)
        
        
        return self.add_layer([y, residual])

# 2. DownBlock
def down_block(x, skips, width, block_depth):
    for _ in range(block_depth):
        x = ResidualBlock(width)(x)
        skips.append(x)
    x = layers.AveragePooling2D(pool_size=2)(x)
    return x

# 3. UpBlock
def up_block(x, skips, width, block_depth):
    x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
    for _ in range(block_depth):
        x = layers.Concatenate()([x, skips.pop()])
        x = ResidualBlock(width)(x)
    return x

# 4. get_networ
def get_network(image_size, widths, block_depth, text_embedding_dim=512):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))
    text_embeddings = keras.Input(shape=(text_embedding_dim,))

    e = layers.Lambda(sinusoidal_embedding, output_shape=(1, 1, 32))(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    t = layers.Dense(32)(text_embeddings)
    t = layers.Activation("swish")(t)
    t = layers.Reshape((1, 1, 32))(t)
    t = layers.UpSampling2D(size=image_size, interpolation="nearest")(t)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e, t])

    skips = []
    
    for width in widths[:-1]:
        x = down_block(x, skips, width, block_depth)

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = up_block(x, skips, width, block_depth)

    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances, text_embeddings], x, name="residual_unet")

class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth, text_encoder):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.text_encoder = text_encoder
        self.network = get_network(image_size, widths, block_depth)
        self.ema_network = get_network(image_size, widths, block_depth)
        self.ema_network.set_weights(self.network.get_weights())
        
    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        # self.kid = KID(name="kid")

    @property
    def metrics(self):
        # return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]
        return [self.noise_loss_tracker, self.image_loss_tracker]
    
    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # TODO: check if the diffusion_times are in the range [0, 1]
        # diffusion_times should be between 0 and 1, corresponding to the start and end of the diffusion process
        diffusion_times = tf.clip_by_value(diffusion_times, 0.0, 1.0)
        
        # angles -> signal and noise rates
        # Use cosine and sine functions to calculate the signal and noise rates based on angles
        start_angle = tf.acos(max_signal_rate)
        end_angle = tf.acos(min_signal_rate)
        t_angle = (1-diffusion_times)*start_angle+diffusion_times*end_angle
        #calculate the noise and signal rates
        noise_rates = tf.sin(t_angle)
        signal_rates = tf.cos(t_angle)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, text_embeddings, training):
        # TODO: implement the denoising network
        # Use the model (self.network or self.ema_network) to predict the noise component in the images
        if training:
            network = self.network
        else:
            network = self.ema_network
        # Predict the noise component and calculate the image component using it
        # Here, use the signal_rate and noise_rate to derive the output components
        # calculate the predicted noises and images
        pred_noises = network([noisy_images, noise_rates, text_embeddings], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps, text_embeddings, guidance_scale=5.5):
        # guidance_scale: 數值越大，越聽文字的話 (建議設 3.0 ~ 7.5)
        
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        next_noisy_images = initial_noise
        
        # 準備一個全零的 embedding (代表無條件)
        uncond_embeddings = tf.zeros_like(text_embeddings)

        for step in range(diffusion_steps):
            diffusion_times = 1.0 - step * step_size
            next_diffusion_times = 1.0 - (step + 1) * step_size
            
            # 製作 shape 正確的時間訊號
            # 注意：這裡需要把 scalar 擴展成 (Batch, 1, 1, 1)
            difussion_times_tensor = tf.ones((num_images, 1, 1, 1), dtype=tf.float32) * diffusion_times
            
            noise_rates, signal_rates = self.diffusion_schedule(difussion_times_tensor)
            next_noise_rates, next_signal_rates = self.diffusion_schedule(tf.ones((num_images, 1, 1, 1), dtype=tf.float32) * next_diffusion_times)

            # [關鍵修改] Classifier-Free Guidance (CFG)
            # 1. 預測有文字條件的噪音
            pred_noises_cond, pred_images_cond = self.denoise(
                next_noisy_images, noise_rates, signal_rates, text_embeddings, training=False
            )
            
            # 2. 預測無文字條件的噪音
            pred_noises_uncond, pred_images_uncond = self.denoise(
                next_noisy_images, noise_rates, signal_rates, uncond_embeddings, training=False
            )
            
            # 3. 混合兩者：將"文字造成的差異"放大 guidance_scale 倍
            pred_noises = pred_noises_uncond + guidance_scale * (pred_noises_cond - pred_noises_uncond)
            pred_images = pred_images_uncond + guidance_scale * (pred_images_cond - pred_images_uncond)

            next_noisy_images = next_signal_rates * pred_images + next_noise_rates * pred_noises

        return pred_images

    def generate(self, num_images, diffusion_steps, input_ids):
        # noise -> images -> denormalized images
        text_embeddings = self.text_encoder(input_ids)
        initial_noise = tf.random.normal(
            shape=(num_images, image_size, image_size, 3)
        )
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps, text_embeddings)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, data):
        input_ids, images = data
        
        # 1. 取得文字 Embedding
        text_embeddings = self.text_encoder(input_ids)
        
        current_batch_size = tf.shape(text_embeddings)[0]
        drop_mask = tf.random.uniform((current_batch_size, 1)) < 0.1
        text_embeddings = tf.where(drop_mask, tf.zeros_like(text_embeddings), text_embeddings)
        
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(batch_size, image_size, image_size, 3))

        # (以下保持不變...)
        diffusion_times = tf.random.uniform(
            shape=(tf.shape(images)[0], 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, text_embeddings, training=True
            )

            noise_loss = self.loss(noises, pred_noises)
            image_loss = self.loss(images, pred_images)

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, data):
        input_ids, images = data
        text_embeddings = self.text_encoder(input_ids)
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(tf.shape(images)[0], image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, text_embeddings, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=batch_size, diffusion_steps=kid_diffusion_steps
        )
        # self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=2, num_cols=4):
        # plot random generated images for visual evaluation of generation quality
        test_ids = sample_input_ids[:num_rows * num_cols]
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=plot_diffusion_steps,
            input_ids=test_ids
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        os.makedirs("samples/demo", exist_ok=True)
        filename = f"samples/demo/train_{epoch}.jpg"
        plt.savefig(filename)
        print(f"\n圖片已儲存至: {filename}")
        plt.close()


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

sample_sentences = [
    "the flower shown has yellow anther red pistil and bright red petals.",
    "this flower has petals that are yellow, white and purple and has dark lines",
    "the petals on this flower are white with a yellow center",
    "this flower has a lot of small round pink petals.",
    "this flower is orange in color, and has petals that are ruffled and rounded.",
    "the flower has yellow petals and the center of it is brown.",
    "this flower has petals that are blue and white.",
    "these white flowers have petals that start off white in color and end in a white towards the tips."
]
encodings = tokenizer(sample_sentences, padding="max_length", truncation=True, max_length=77, return_tensors="tf")
sample_input_ids = encodings['input_ids']

epoch_var = tf.Variable(0, trainable=False, dtype=tf.int64, name="epoch")
text_encoder = ClipTextEncoder()
model = DiffusionModel(image_size, widths, block_depth, text_encoder)
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
    loss=tf.keras.losses.MeanSquaredError(),
)

train_dataset = dataset_generator(data_path + '/text2ImgData.pkl', batch_size)

total_samples = len(pd.read_pickle(data_path + '/text2ImgData.pkl'))
steps_per_epoch = total_samples // batch_size
image_ds = train_dataset.map(lambda text, img: img)
image_ds = image_ds.take(steps_per_epoch)
print("Adapting Normalizer...")
model.normalizer.adapt(image_ds)
print("Adapt Complete.")

checkpoint = tf.train.Checkpoint(model=model,
                                 epoch_var=epoch_var)
manager = tf.train.CheckpointManager(checkpoint, './ckpts',  max_to_keep=20,
                                     checkpoint_name='diffusion')

def save_checkpoint_func(epoch, logs):
    assert epoch==int(epoch_var.numpy())
    epoch_var.assign_add(1)
    save_path = manager.save()
    print("\nSaved checkpoint for epoch {}: {}".format(int(epoch_var), save_path))

# 建立 callback
checkpoint_callback = keras.callbacks.LambdaCallback(
    on_epoch_end=save_checkpoint_func
)

if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
    print(f"Restored from {manager.latest_checkpoint}")
else:
    print("Initializing from scratch")

print(f"Start training with {steps_per_epoch} steps per epoch.")
model.fit(
    train_dataset,
    initial_epoch=int(epoch_var.numpy()),
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    callbacks=[
        keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
        checkpoint_callback
    ],
)
