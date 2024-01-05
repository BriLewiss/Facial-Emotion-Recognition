# Emotional-Monitoring
This repository contains code for a Facial Emotion Recognition (FER) system implemented using Convolutional Neural Networks (CNNs). The FER system is designed to recognize four emotions: anger, happy, sad, and neutral. The code utilizes the Keras library with a TensorFlow backend for building and training the CNN model.

**Dataset**
The FER dataset used in this project is loaded from a CSV file (fer2013.csv) and preprocessed to remove certain emotions, leaving only the emotions of interest (anger, happy, sad, and neutral). The dataset preprocessing is essential to ensure that the model learns to recognize these specific emotions.

**Model Architecture**
The CNN model architecture consists of several layers:

1st Convolution Layer with 160 units, followed by Batch Normalization and ReLU activation.
2nd Convolution Layer with 64 units, followed by Batch Normalization and ReLU activation.
3rd Convolution Layer with 224 units, followed by Batch Normalization and ReLU activation.
Fully connected Dense Layer with 512 units and ReLU activation.
Output Layer with softmax activation for classifying emotions.
Dropout layers are used for regularization to prevent overfitting.

**Training**
The model is trained using the preprocessed dataset with a batch size of 64 and for 30 epochs. An early stopping callback is implemented to monitor validation loss and stop training if it stops improving.

**Video Emotion Recognition**
Included in this repository is code for real-time video emotion recognition using the trained model. The code captures video from your webcam and overlays predicted emotions on detected faces.
