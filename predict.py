from config import config as C
from config import functions
from config.model_architecture import CNN
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

# PREPROCESSING
# Read dataset
val_data = pd.read_csv(C.DATASET + 'train.csv')
test_data = pd.read_csv(C.DATASET + 'test.csv')

# Separate features from labels
y_val = val_data.iloc[:, 0]
X_val = val_data.iloc[:, 1:]
X_test = test_data

# Rescale X to [0, 1]
X_val = X_val / 255
X_test = X_test / 255

# Reshape X into [n, 28, 28, channels]
X_val = np.array(X_val).reshape(X_val.shape[0], 28, 28, 1)
X_test = np.array(X_test).reshape(X_test.shape[0], 28, 28, 1)

# One-hot encode labels
lb = LabelBinarizer()
y_val = lb.fit_transform(y_val)

filepaths = []
for class_num in range(10):
    filepath = f'dataset/{class_num}.csv'
    filepaths.append(filepath)

train_set = functions.csv_reader_dataset(filepaths)

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

# MODEL
# Compile model
opt = SGD(learning_rate=C.LEARNING_RATE)
model = CNN.build(width=28, height=28, depth=1, classes=10)

model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

# Construct callbacks
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=10)
callbacks = [early_stopping, lr_scheduler]

# Train model
H = model.fit(
    train_set,
    validation_data=(X_val, y_val),
    batch_size=C.BATCH_SIZE,
    epochs=C.EPOCHS,
    callbacks=callbacks,
    verbose=1
)
print('Training done!')


# PREDICTIONS
# Load best model and predict test data
y_test = model.predict(X_test)
pred = y_test.argmax(axis=1)

# Save to csv in required format
df = pd.DataFrame({
    'ImageID': np.arange(1,28001),
    'Label': pred
})
df.to_csv(C.OUTPUT + 'kaggle_predictions_gan.csv', index=False)
print('Predictions saved!')


# Save model
from tensorflow.keras.utils import plot_model

plot_model(model, show_shapes=True, to_file=C.MODEL + 'model.png')
model.save(C.MODEL + 'best_model.h5')
C

import matplotlib.pyplot as plt

# PLOT
plt.style.use('ggplot')
plt.figure()
for y in ['loss', 'val_loss', 'accuracy', 'val_accuracy']:
    plt.plot(np.arange(1, len(H.history[y]) + 1), H.history[y], label=y)
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig(C.OUTPUT + 'loss_acc_plot.jpg')
plt.close()
print('Model saved!')