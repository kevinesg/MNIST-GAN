from GAN import train_gan
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

BATCH_SIZE = 32
CODINGS_SIZE = 100
CLASS_SIZE = 10000  # This is the number of new data points per class (10 classes)

train = pd.read_csv('dataset/train.csv')
train_copy = train.copy()
X_train = train.iloc[:, 1:]
X_train = np.array(X_train).reshape([-1, 28, 28, 1])

# Loop over each of the 10 classes
for class_num in range(10):
    # Apply masking
    data = X_train[train_copy.iloc[:, 0]==class_num]
    # Rescale values
    data =  data / 255
    # Prepare data for training
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(data.shape[0])
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(1)

    # Instantiate the GAN layers
    generator = keras.models.Sequential([
        keras.layers.Dense(7 * 7 * 128, input_shape=[CODINGS_SIZE]),
        keras.layers.Reshape([7, 7, 128]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same',
            activation='selu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding='same',
            activation='sigmoid'),
    ])

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

    gan = keras.models.Sequential([
        generator,
        discriminator
    ])

    # Compile the discriminator and whole GAN
    discriminator.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop'
    )

    # Freeze discriminator layers so that only generator will be trained
    discriminator.trainable = False
    gan.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop'
    )

    # Train the GAN
    print(f'[INFO] Training GAN for class #{class_num}...')
    train_gan(gan, dataset, BATCH_SIZE, CODINGS_SIZE)

    # Generate synthetic data
    noise = np.random.uniform(0, 1, size=(CLASS_SIZE, CODINGS_SIZE))
    gen_imgs = generator.predict(noise)
    gen_imgs = gen_imgs.reshape([gen_imgs.shape[0], 784])
    labels = np.full((gen_imgs.shape[0], 1), class_num)
    new_data = np.hstack([labels, gen_imgs])
    new_data = pd.DataFrame(new_data, columns=train.columns)
    # Add the new synthetic data to existing data
    train = pd.concat([train, new_data])

# Save new dataset
print('[INFO] Saving dataset...')
train.to_csv('dataset/new_dataset.csv', index=False)
print('[INFO] Done!')
