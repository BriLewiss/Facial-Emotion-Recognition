import sys, os  
import pandas as pd  
import numpy as np  
from keras.models import Sequential  
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D  
from keras.losses import categorical_crossentropy  
from keras.optimizers import Adam  
from keras.regularizers import l2  
from keras.utils import to_categorical 
from keras.regularizers import l1
from keras.callbacks import EarlyStopping


# pd.set_option('display.max_rows', 500)  
# pd.set_option('display.max_columns', 500)  
# pd.set_option('display.width', 1000)  
  
df=pd.read_csv('/Users/macbook/Documents/evelopmet/Datasets/fer2013.csv')  

# Assuming 'disgust', 'fear', and 'surprise' are represented by 1, 2, 5 respectively
df = df[~df['emotion'].isin([1, 2, 5])]

  
# print(df.info())  
# print(df["Usage"].value_counts())  
  
# print(df.head())  
X_train,train_y,X_test,test_y=[],[],[],[]  
  
for index, row in df.iterrows():  
    val=row['pixels'].split(" ")  
    try:  
        if 'Training' in row['Usage']:  
           X_train.append(np.array(val,'float32'))  
           train_y.append(row['emotion'])  
        elif 'PublicTest' in row['Usage']:  
           X_test.append(np.array(val,'float32'))  
           test_y.append(row['emotion'])  
    except:  
        print(f"error occured at index :{index} and row:{row}")  

# Old to new label mapping
label_mapping = {0: 0, 3: 1, 4: 2, 6: 3}

# Update train_y and test_y with new labels
train_y = np.array([label_mapping[label] for label in train_y])
test_y = np.array([label_mapping[label] for label in test_y])

  
  
num_features = 64  
num_labels = 4  
batch_size = 64  
epochs = 30  
width, height = 48, 48  
  
  
X_train = np.array(X_train,'float32')  
train_y = np.array(train_y,'float32')  
X_test = np.array(X_test,'float32')  
test_y = np.array(test_y,'float32')  

# Check the range of train_y
print("Min label:", train_y.min(), "Max label:", train_y.max())
  
train_y=to_categorical(train_y, num_classes=num_labels)  
test_y=to_categorical(test_y, num_classes=num_labels)  
  
#cannot produce  
#normalizing data between oand 1  
X_train -= np.mean(X_train, axis=0)  
X_train /= np.std(X_train, axis=0)  
  
X_test -= np.mean(X_test, axis=0)  
X_test /= np.std(X_test, axis=0)  
  
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)  
  
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)  
  
# print(f"shape:{X_train.shape}")  
##designing the cnn  
model = Sequential()

# 1st convolution layer with 160 units
model.add(Conv2D(160, (3, 3), padding='same', input_shape=(width, height, 1), kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))  # Dropout rate of 0.2

# 2nd convolution layer with 64 units
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))  # Using the same dropout rate for consistency

# 3rd convolution layer with 224 units
model.add(Conv2D(224, (3, 3), padding='same', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# No dropout after this layer

model.add(Flatten())

# Fully connected dense layer with 512 units
model.add(Dense(512, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))  # Dropout rate of 0.4

# Output layer
model.add(Dense(num_labels, activation='softmax'))

# Compile the model with a learning rate of 0.00010222
model.compile(loss=categorical_crossentropy, 
              optimizer=Adam(lr=0.00010222), 
              metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min')
  
#Training the model  
model.fit(X_train, 
          train_y, 
          batch_size=batch_size, 
          epochs=epochs, 
          verbose=1, 
          validation_data=(X_test, test_y), 
          callbacks=[early_stopping])

  
  
#Saving the  model to  use it later on  
fer_json = model.to_json()  
with open("fer2.json", "w") as json_file:  
    json_file.write(fer_json)  
model.save_weights("fer2.h5")  
