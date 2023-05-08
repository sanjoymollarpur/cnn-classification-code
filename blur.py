from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf
import pandas as pd
print(tf.__version__)
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Model
import seaborn as sns 

train_path = 'data/train'
valid_path = 'data/test'



epochs=80
IMAGE_SIZE = [224, 224]
batch_size=16


vgg16_base = VGG16(include_top=False,input_tensor=None, input_shape=(224, 224, 3))

# In[10]:

print('Adding new layers...')
output = vgg16_base.get_layer(index = -1).output  
output = Flatten()(output)
# let's add a fully-connected layer
output = Dense(1024,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.2)(output)
output = Dense(1024,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.2)(output)
output = Dense(3, activation='softmax')(output)
print('New layers added!')

# In[11]:

vgg16_model = Model(vgg16_base.input, output)
for layer in vgg16_model.layers[:-7]:
    layer.trainable = False

print(vgg16_model.summary())


model=vgg16_model




# tf.optimizers.Adam(lr=0.001)
model.compile(
  loss='categorical_crossentropy',
  optimizer=tf.optimizers.Adam(lr=0.0001),
  metrics=['accuracy']
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/merge/1723-320-329/train',
                                                 target_size = (224, 224),
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical',
                                                 shuffle=True)
                                                


test_set = test_datagen.flow_from_directory('data/merge/1723-320-329/test',
                                            target_size = (224, 224),
                                            batch_size = batch_size,
                                            class_mode = 'categorical',
                                            shuffle=True)


r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=epochs,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

from tensorflow.keras.models import load_model

model.save('model/blur.h5')

import matplotlib.pyplot as plt

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
# plt.show()
plt.grid()
plt.savefig('LossVal_loss')
plt.close()

plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.grid()
plt.savefig('AccVal_acc')
plt.close()



import matplotlib.pyplot as plt 
import seaborn as sns

import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming you have already trained a model and have a test dataset
# Replace the path with the actual path to your test dataset
test_directory = 'data/new-300-2-classes/test'
test_directory = "data/merge/1723-320/test"

# Create an ImageDataGenerator for the test dataset
test_datagen = ImageDataGenerator(rescale=1./255)

# Prepare the data generator
test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=(224, 224),
    batch_size=1,
    class_mode='sparse',
    shuffle=False
)

# Obtain the true labels and predicted labels
true_labels = test_generator.classes
predictions = model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion matrix:\n", conf_matrix)


plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.savefig("cm-matrix-3.png")
# plt.show()
plt.close()
