# Documentation
This is the official documentation of the Singularity Machine learning framework

-------------------------------------------------------------------------------------

## Input layer (Universal input layer)

  ``` python
  Input_layer(Input_data, Input_shape, Out_shape)
  ```
  
  (Array/List)   ```Input_data``` is the training/validation data that is being passed. From there the input shape is also derived. Every array/list should have the        same shape.

  (Array) ```Input_shape``` is the shape the data has.

  (Array) ```Out_shape``` is the shape the output should be when forward propagation was passed.

-------------------------------------------------------------------------------------

## Adding fully-connected layers

``` python
add_layer(neurons, activation_function)
```

(Integer) ```neurons``` are the number of neurons the layer is going to have.

(activation_function) ```activation_function``` are the activation functions that are used in that layer (e.g. ReLU, GeLU).

-------------------------------------------------------------------------------------

## Training

``` python
train(X_train, y_train, learning_rate, epochs, batch_size)
```

(Array) ```X_train``` is the data on which the network will be trained on

(Array) ```y_train``` are the labels corresponding to the data.

(Float) ```learning_rate``` if none then default set to 0.01.

(Integer) ```epochs``` are iterations of training the network goes through.

(Integer) ```batch_size``` is the batch that is being used from the whole training pool for each of the ```epochs```.

-------------------------------------------------------------------------------------

## Evaluate (With trained network) - returns the accuracy of the network

``` python
Evaluate(X_test, y_test)
```

(Array) ```X_test``` is the data that is given the Network for test purposes.

(Array) ```y_test``` are the labels, which need to be one-hot encoded.

-------------------------------------------------------------------------------------

## Augmentation (to increase dataset size) - only for image datasets

  ``` python
  augmentor(DATADIR, CATEGORIES, FLIP, NOISE, ROTATE)
  ```
  (String) ```DATADIR``` is the path of the folder that contains the folders containing the images.

  Example structure:
  ```
  DATADIR
  ├── Dog
  │   └── dog_example.png
  └── Cat
      └── cat_example.png
  ```

  (List) ```CATEGORIES``` is a list with the names of the folders containing the images (e.g. ```Dog```, ```Cat```).

  (Boolean) ```FLIP```, when activated flips every image and saves it in the same directory.

  (Boolean) ```NOISEE```, when activated adds Noise to every image and saves it in the same directory.

  (Boolean) ```ROTATE```, when activated rotates every image and saves it in the same directory.

-------------------------------------------------------------------------------------

## Image preprocessing (to convert images in arrays) - only for image datasets

  ``` python
  img_preprocessing(DATADIR, CATEGORIES, IMG_SIZE)
  ```

  (String) ```DATADIR``` is the path of the folder that contains the folders containing the images.

  Example structure:
  ```
  DATADIR
  ├── Dog
  │   └── dog_example.png
  └── Cat
      └── cat_example.png
  ```

  (List) ```CATEGORIES``` is a list with the names of the folders containing the images (e.g. ```Dog```, ```Cat```).

  (Integer) ```IMG_SIZE``` defines the size which all images are automatically reshaped to.

-------------------------------------------------------------------------------------

## Tokenizer - only for strings

``` python
tokenizer(Input_string)
```

(String) ```Input_string``` is the list which should be tokenized and vectorized according to the word array's indicies.
