import GAN
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

BATCH_SIZE = 128
EPOCHS = 200
CODINGS_SIZE = 100
CLASS_SIZE = 100000  # This is the number of new data points per class (10 classes)

train = pd.read_csv('dataset/train.csv')
train_copy = train.copy()
X_train = train.iloc[:, 1:]
X_train = np.array(X_train).reshape([-1, 28, 28, 1])

# Loop over each of the 10 classes
for class_num in range(10):
    # Apply masking
    data = X_train[train_copy.iloc[:, 0]==class_num]
    # Rescale values
    data =  data / 255 * 2 - 1
    # Prepare data for training
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(data.shape[0])
    dataset = dataset.batch(int(BATCH_SIZE / 2), drop_remainder=True).prefetch(1)

    # Instantiate the GAN layers
    generator = GAN.generator(CODINGS_SIZE)
    discriminator = GAN.discriminator()

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
    GAN.train_gan(gan, dataset, BATCH_SIZE, CODINGS_SIZE, class_num, EPOCHS)

    # Generate synthetic data
    noise = np.random.uniform(0, 1, size=(CLASS_SIZE, CODINGS_SIZE))
    # Rescale the generator output from [-1, 1] to [0, 1]
    gen_imgs = generator.predict(noise) / 2 + 0.5
    # Flatten each image 2D array
    gen_imgs = gen_imgs.reshape([gen_imgs.shape[0], 784])
    # Generate a 1D array of labels
    labels = np.full((gen_imgs.shape[0], 1), class_num)
    new_data = np.hstack([labels, gen_imgs])
    new_data = pd.DataFrame(new_data, columns=train.columns)
    new_data.iloc[:, 0] = new_data.iloc[:, 0].astype(np.int8)
    # Save new dataset
    print('[INFO] Saving new dataset...')
    new_data.to_csv(f'dataset/{class_num}.csv', index=False)

print('[INFO] All done!')
