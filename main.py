# coding: utf-8
import os
import pathlib

import cv2
import numpy as np
from keras import layers
from keras import models
from keras.optimizers import Adam


def _load_data(path):
    imgs = []
    for im in os.listdir(path)[:4096]:
        im = os.path.join(path, im)
        im = cv2.imread(im)
        im = cv2.resize(im, (64, 64))
        imgs.append(im)
    return imgs


def load_data(path='../faces/', fn='./faces.npy'):
    if not os.path.isfile(fn):
        imgs = _load_data(path)
        np.save(fn, imgs)
    imgs = np.load(fn)
    return imgs


x_train = load_data()
x_train = x_train.astype(np.float32)
x_train = x_train / 127.5 - 1.0
print(x_train.shape)

# 一些参数
latent_dim = 100
iterations = 3 * 60 * 60
batch_size = 128
save_dir = './faces/'
h, w, c = 64, 64, 3

generator = models.Sequential([
    layers.Reshape((1, 1, latent_dim), input_shape=(latent_dim,)),
    layers.Conv2DTranspose(512, 4),
    layers.BatchNormalization(momentum=0.5),
    layers.LeakyReLU(0.2),
    layers.Conv2DTranspose(256, 4, strides=2, padding='same'),
    layers.BatchNormalization(momentum=0.5),
    layers.LeakyReLU(0.2),
    layers.Conv2DTranspose(128, 4, strides=2, padding='same'),
    layers.BatchNormalization(momentum=0.5),
    layers.LeakyReLU(0.2),
    layers.Conv2DTranspose(64, 4, strides=2, padding='same'),
    layers.BatchNormalization(momentum=0.5),
    layers.LeakyReLU(0.2),
    layers.Conv2D(64, 3, padding='same'),
    layers.BatchNormalization(momentum=0.5),
    layers.LeakyReLU(0.2),
    layers.Conv2DTranspose(c, 4, strides=2, activation='tanh', padding='same'),
])

discriminator = models.Sequential([
    layers.Conv2D(64, 4, strides=2, padding='same', input_shape=(h, w, c)),
    layers.LeakyReLU(0.2),
    layers.Conv2D(128, 4, strides=2, padding='same'),
    layers.BatchNormalization(momentum=0.5),
    layers.LeakyReLU(0.2),
    layers.Conv2D(256, 4, strides=2, padding='same'),
    layers.BatchNormalization(momentum=0.5),
    layers.LeakyReLU(0.2),
    layers.Conv2D(512, 4, strides=2, padding='same'),
    layers.BatchNormalization(momentum=0.5),
    layers.LeakyReLU(0.2),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid'),
])
discriminator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                      loss='binary_crossentropy')

discriminator.trainable = False
gan = models.Sequential([
    generator,
    discriminator,
])
gan.compile(optimizer=Adam(lr=0.00015, beta_1=0.5),
            loss='binary_crossentropy')

generator.summary()
discriminator.summary()
gan.summary()

random_vectors = np.random.normal(size=(10 * 10, latent_dim))
imgs = np.zeros((10 * h, 10 * h, c), dtype=np.uint8)

# Assemble labels that say "all real images"
misleading_targets = np.ones((batch_size, 1))

# Start training loop
pathlib.Path(save_dir).mkdir(exist_ok=True)
start = 0
for step in range(iterations):
    # Sample random points in the latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    # Decode them to fake images
    generated_images = generator.predict(random_latent_vectors)

    # Combine them with real images
    stop = start + batch_size
    real_images = x_train[start: stop]

    # Train the discriminator
    labels = np.ones((batch_size, 1)) - np.random.random_sample((batch_size, 1)) * 0.2
    d_loss = discriminator.train_on_batch(real_images, labels)
    labels = np.random.random_sample((batch_size, 1)) * 0.2
    d_loss = discriminator.train_on_batch(generated_images, labels)

    # Train the generator (via the gan model,
    # where the discriminator weights are frozen)
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0

    # Occasionally save / plot
    if step % 100 == 0:
        # Save model weights
        gan.save('gan.h5', include_optimizer=False)

        # Print metrics
        print('discriminator loss at step %s: %s' % (step, d_loss))
        print('adversarial loss at step %s: %s' % (step, a_loss))

        # Save one generated image
        generated_images = generator.predict(random_vectors)
        generated_images += 1.0
        generated_images *= 127.5
        for i in range(10):
            for j in range(10):
                imgs[i * h:i * h + h, j * h:j * h + h] = generated_images[i * 10 + j]
        cv2.imwrite(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'), imgs)
