# MNIST data generator using GAN
Run `generate_data.py` to generate new data. You can adjust the `CLASS_SIZE` (number of generated data points per class). I used Deep Convolutional GAN (DCGAN) here since the dataset is composed of images. Samples of synthetic images are produced by the GAN after training. You can see below some sample synthetic images produced by the GAN after 200 epochs.

![image](https://user-images.githubusercontent.com/60960803/127313801-aa8ecaff-4d51-44e7-b06a-54942dc0d713.png)

As you can see, there are a lot of noise and some images are so distorted that they should be removed from the dataset. To check the quality of the generated images, I used `predict.py` to train a CNN using only the generated images. I used the original training set provided by Kaggle as the validation set. The CNN architecture I used gave a 99.5% accuracy on the test set (using the provided training dataset).

![loss_acc_plot](https://user-images.githubusercontent.com/60960803/127314113-44dc9946-6a8f-4d56-943d-fb43225341d9.jpg)

Here is the loss and accuracy vs number of epochs plot. As you can see, the model pretty much acheived overfitting after a couple of epochs. My guess is it's because there's not enough variety in the nature of the dataset (binary images of handwritten numbers). Take note that I used a total of 1 million generated images as my training set, compared the original training set from Kaggle which is only composed of 50,000 images. Training on the generated images gave ~75% validation accuracy and Kaggle gave a 76% accuracy for the test set.

The results aren't ideal but considering I didn't use a single image from the original dataset, 76% accuracy is not bad. The area of improvement is obviously the DCGAN architecture.
#
I didn't upload the csv files of the generated images because the total size is 16GB but you can generate your own csv files by running the code here.

If you have any questions or suggestions, please let me know. Thanks for reading!
