#https://www.youtube.com/watch?v=j-3vuBynnOE
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import csv
 

data_file = input("Enter path to data file: ")
CATEGORIES = ["Cat", "Dog"]

for category in CATEGORIES:
    path = os.path.join(data_file, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (100, 100))
        plt.imshow(new_array, cmap = "gray")
        plt.show()
        break
    break

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(data_file, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (100, 100))
                training_data.append([new_array, class_num])
            except Exception as  e:
                pass

create_training_data()
#print(len(training_data))
"""
import random
random.shuffle(training_data)

for sample in training_data:
    print(sample[1])
"""
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)    
    
X = np.array(X).reshape(-1, 100, 100, 1)



pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


pickle_in = open("X.pickle","rb" )
X = pickle.load(pickle_in)


X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = X/255.0
y = np.array(y)

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])


model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)

csv_file = "model_details.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model Name", "Architecture", "Epochs"])
    #writer.writerows(model_details)

print(f"All model details have been saved to {csv_file}. Thank you for using this program!")



































