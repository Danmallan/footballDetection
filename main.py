import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
Sequential = tensorflow.keras.Sequential
layers = tensorflow.keras.layers
models = tensorflow.keras.models
ImageDataGenerator = tensorflow.keras.preprocessing.image
MaxPooling2D = tensorflow.keras.layers.MaxPooling2D
Conv2D = tensorflow.keras.layers.Conv2D
Flatten = tensorflow.keras.layers.Flatten
datasets = tensorflow.keras.datasets
Dense = tensorflow.keras.layers.Dense
Dropout = tensorflow.keras.layers.Dropout
SGD = tensorflow.keras.optimizers.legacy.SGD
MaxNorm = tensorflow.keras.constraints.MaxNorm
to_categorical = tensorflow.keras.utils.to_categorical
from keras.preprocessing.image import load_img, img_to_array

# Load train annotations.csv
annotations = pd.read_csv('/Users/sdanmallan/Downloads/soccer_exercise/train/annotations.csv')

# Preprocess data
images = []
coordinates = []

for index, row in annotations.iterrows():
    image_path = '/Users/sdanmallan/Downloads/soccer_exercise/train/' + row['filename']
    image = load_img(image_path, target_size=(224, 224))  # Adjust target_size as needed
    image = img_to_array(image) / 255.0  # Normalize pixel values
    images.append(image)

    xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
    coordinates.append([xmin, ymin, xmax, ymax])

images = np.array(images)
coordinates = np.array(coordinates)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(224, activation='softmax'))
model.add(Dense(4))  # Output layer for xmin, ymin, xmax, ymax coordinates

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

history = model.fit(images, coordinates, epochs=10, batch_size=32, validation_split=0.2)

print(history.history['accuracy'])
print(history.history['val_accuracy'])

# Load test dataset
test_annotations = pd.read_csv('/Users/sdanmallan/Downloads/soccer_exercise/test/annotations.csv')
test_images = []
test_coordinates = []

for index, row in test_annotations.iterrows():
    test_image_path = '/Users/sdanmallan/Downloads/soccer_exercise/test/' + row['filename']
    test_image = load_img(test_image_path, target_size=(224, 224))
    test_image = img_to_array(test_image) / 255.0
    test_images.append(test_image)

    test_xmin, test_ymin, test_xmax, test_ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
    test_coordinates.append([test_xmin, test_ymin, test_xmax, test_ymax])

test_images = np.array(test_images)
test_coordinates = np.array(test_coordinates)

# Evaluate model
loss, accuracy = model.evaluate(test_images, test_coordinates)
print('Test Accuracy:', accuracy)

# Plot Accuracy Chart
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
