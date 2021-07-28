import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2


def generator(CODINGS_SIZE):
    generator = keras.models.Sequential([
        keras.layers.Dense(7 * 7 * 128, input_shape=[CODINGS_SIZE]),
        keras.layers.Reshape([7, 7, 128]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', activation='selu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding='same', activation='tanh')
    ])

    return generator


def discriminator():
    discriminator = keras.models.Sequential([
        keras.layers.Conv2D(64, kernel_size=5, strides=2, padding='same',
            activation=keras.layers.LeakyReLU(0.2), input_shape=[28, 28, 1]),
        keras.layers.Dropout(0.4),
        keras.layers.Conv2D(128, kernel_size=5, strides=2, padding='same',
            activation=keras.layers.LeakyReLU(0.2)),
        keras.layers.Dropout(0.4),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    return discriminator


def train_gan(gan, dataset, batch_size, codings_size, class_num, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print(f'[INFO] Epoch #{epoch + 1}')
        for X_batch in dataset:
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[int(batch_size / 2), codings_size])
            generated_images = generator.predict(noise)
            discriminator.trainable = True  # Just to ignore keras warning
            discriminator.train_on_batch(X_batch, tf.constant(np.ones([int(batch_size / 2), 1])))
            discriminator.train_on_batch(
                generated_images, tf.constant(np.zeros([int(batch_size / 2), 1])))

            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant(np.ones([batch_size, 1]))
            discriminator.trainable = False # Just to ignore keras warning
            gan.train_on_batch(noise, y2)

    # Optional
    # Print 20 samples after each class to check quality
    noise = tf.random.normal(shape=(20, codings_size))
    gen_imgs = generator.predict(noise) / 2 + 0.5
    line1 = np.hstack([gen_imgs[0], gen_imgs[1], gen_imgs[2], gen_imgs[3],gen_imgs[4]])
    line2 = np.hstack([gen_imgs[5], gen_imgs[6], gen_imgs[7], gen_imgs[8], gen_imgs[9]])
    line3 = np.hstack([gen_imgs[10], gen_imgs[11], gen_imgs[12], gen_imgs[13], gen_imgs[14]])
    line4 = np.hstack([gen_imgs[15], gen_imgs[16], gen_imgs[17], gen_imgs[18], gen_imgs[19]])
    imgs = np.vstack([line1, line2, line3, line4]) * 255
    imgs = cv2.resize(imgs, (700, 560))
    cv2.imwrite(f'dataset/sample_images_{n_epochs}epochs/{class_num}.jpg', imgs)
    #cv2.imshow(f'{class_num}', imgs)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()