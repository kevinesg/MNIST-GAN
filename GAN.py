import numpy as np
import cv2

def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print(f'[INFO] Epoch #{epoch + 1}')
        for X_batch in dataset:
            # phase 1 - training the discriminator
            noise = np.random.normal(size=[batch_size, codings_size])
            generated_images = generator.predict(noise)
            X_fake_and_real = np.vstack([generated_images, X_batch])
            y1 = np.vstack([np.zeros([batch_size, 1]), np.ones([batch_size, 1])])
            discriminator.trainable = True  # Just to ignore keras warning
            discriminator.train_on_batch(X_fake_and_real, y1)

            # phase 2 - training the generator
            noise = np.random.normal(size=[batch_size, codings_size])
            y2 = np.ones([batch_size, 1])
            discriminator.trainable = False # Just to ignore keras warning
            gan.train_on_batch(noise, y2)


    # Optional
    # Print 20 samples after each class to check quality
    noise = np.random.uniform(0, 1, size=(20, codings_size))
    gen_imgs = generator.predict(noise)
    line1 = np.hstack([gen_imgs[0], gen_imgs[1], gen_imgs[2], gen_imgs[3], gen_imgs[4]])
    line2 = np.hstack([gen_imgs[5], gen_imgs[6], gen_imgs[7], gen_imgs[8], gen_imgs[9]])
    line3 = np.hstack([gen_imgs[10], gen_imgs[11], gen_imgs[12], gen_imgs[13], gen_imgs[14]])
    line4 = np.hstack([gen_imgs[15], gen_imgs[16], gen_imgs[17], gen_imgs[18], gen_imgs[19]])
    imgs = np.vstack([line1, line2, line3, line4])
    cv2.imshow('', cv2.resize(imgs, (700, 560)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()