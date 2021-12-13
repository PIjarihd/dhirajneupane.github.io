---
layout: post
title: Classification using CNN for Fashion-MNIST Data
excerpt: "CNN classification  Fashion-MNIST"
modified: 2021-12-12
tags: [machine learning]
comments: true
category: blog
---
Today I am going to use CNN to classify Fashion-MNIST dataset. MNIST Fashion dataset contains total 70,000 images and the respective laberls. Out of them, 60000 images are training images and the rest 10,000 are test images.
  Let's get started.
  
### Importing the necessary libraries
```python
  # Import necessary libraries
  import tensorflow as tf
  import numpy as np
  import matplotlib.pyplot as plt
  from tensorflow import keras
```

## Loading the dataset
  
  Let's load the fashion-MNIST dataset and also load the train-test images and labels
  ```python
  mnist=tf.keras.dataset.fashion_mnist
  (training_images,training_labels),(test_images,test_labels)=mnnist.load_data()
  ```
  
  The dataset is loaded and training and testing images are also split. Let's visualize the first image.
  ```python
  np.set_printoptions(linewidth=200) #This function determines the way floating point numbers, arrays and other NumPy objects are displayed.
  #Linewidth is the number of characters per line for the purpose of inserting line breaks
  plt.imshow(training_images[0]) # Dispalys the first image of the training set
  print(training_images[0]) #Prints the 28*28 pixel values of the first image
  print(training_labels[0] #Prints the label for the first image
  ```
  
### Normalizing the data
  Neural network work better with the normalized data. The pixel values are from 0-255 which will be normalized from 0-1.
 ```python
  training_images=training_images/255.0
  test_images=test_images/255.0
  ```
  
### Designing Model, training the model and evaluation
  ```python
  model=tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(128,activation=tf.nn.relu),
                        tf.keras.layers.Dense(10,activation=tf.nn.softmax)])
  
  model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
  model.fit(training_images, training_labels, epochs=200)
  
  model.evaluate(test_images,test_labels)
  ```
Sequential: That defines a Sequence of layers in the neural network

Flatten: Flattenning just takes that square and turns it into a 1 dimensional set.

Dense: Adds a layer of neurons

Each layer of neurons need an activation function to tell them what to do. There's lots of options, but just use these for now.
  
  
  
  
  
  
  
  
