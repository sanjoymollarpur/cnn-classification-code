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


vgg16_base = VGG16(include_top=False,
                   input_tensor=None, input_shape=(224, 224, 3))

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
output = Dense(2, activation='softmax')(output)
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

training_set = train_datagen.flow_from_directory('data/merge/1723-320-2/train',
                                                 target_size = (224, 224),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')
                                                


test_set = test_datagen.flow_from_directory('data/merge/1723-320-2/test',
                                            target_size = (224, 224),
                                            batch_size = 16,
                                            class_mode = 'categorical')


r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=epochs,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

from tensorflow.keras.models import load_model

model.save('model/blur-2.h5')

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


y_pred = model.predict(test_set)
import numpy as np
y_pred = np.argmax(y_pred, axis=1)

target_names = []
for key in training_set.class_indices:
    target_names.append(key)

y_true=test_set.classes
cm = confusion_matrix(y_true, y_pred)

cm_df = pd.DataFrame(cm,
                     index = target_names, 
                     columns = target_names)

plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.savefig("cm-matrix.png")
# plt.show()
plt.close()

print(cm_df)





