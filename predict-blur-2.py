
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model=load_model('model/blur-2.h5')
import numpy as np

import glob 
import cv2
# c=0
# import time
# for i in glob.glob("data/new-300/test/blur-1/*.jpg"):
#     # time.sleep(1)
#     img = image.load_img(i, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     # print(x.shape)
#     images = np.vstack([x])
#     classes = model.predict(x)
#     # print(classes)
#     y_pred = np.argmax(classes, axis=1)
#     if y_pred[0]==0:
#         c+=1
#     # print(y_pred)
#     # print(i)
# print("cat", c)


c1=0
import time
for i in glob.glob("data/merge/1723-320-2/test/blur-2/*.jpg"):
    # time.sleep(1)
    test_image = image.load_img(i, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0

    # Make the prediction
    classes = model.predict(test_image)
    y_pred = np.argmax(classes, axis=1)
    if y_pred[0]==0:
        c1+=1
    # print(y_pred)
    # print(i)


c2=0
import time
for i in glob.glob("data/merge/1723-320-2/test/clear/*.jpg"):
    # time.sleep(1)
    # img = image.load_img(i, target_size=(224, 224))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # # print(x.shape)
    # images = np.vstack([x])
    # classes = model.predict(x)
    test_image = image.load_img(i, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0

    # Make the prediction
    classes = model.predict(test_image)
    # print(classes)
    y_pred = np.argmax(classes, axis=1)
    if y_pred[0]==1:
        c2+=1


print("dog", c1, c2)


import matplotlib.pyplot as plt 
import seaborn as sns

import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming you have already trained a model and have a test dataset
# Replace the path with the actual path to your test dataset
test_directory = 'data/new-300-2-classes/test'
test_directory = "data/merge/1723-320-2/test"

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
