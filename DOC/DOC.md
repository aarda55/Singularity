# Documentation
## This is the official documentation of the Singularity Machine learning framework


<br><br>

### Content includes:

1. [Single model functions](#singlemodel)
2. [auxiliary functions](#auxiliary)
3. [cross framework functions](#cross)
4. [MulliModel functions](#multimodel)

-------------------------------------------------------------------------------------

#### Input layer (Universal input layer) <a name="singlemodel"></a>

  ``` python
  input_layer(Input_data, Input_shape, Out_shape)
  ```
  
  (Array/List)   ```Input_data``` is the training/validation data that is being passed. From there the input shape is also derived. Every array/list should have the        same shape.

  (Array) ```Input_shape``` is the shape the data has.

  (Array) ```Out_shape``` is the shape the output should be when forward propagation was passed.

-------------------------------------------------------------------------------------

#### Adding fully-connected layers <a name="singlemodel"></a>

``` python
add_layer(Neurons, Activation_function)
```

(Integer) ```neurons``` are the number of neurons the layer is going to have.

(activation_function) ```activation_function``` are the activation functions that are used in that layer (e.g. ReLU, GeLU).

-------------------------------------------------------------------------------------

#### Training <a name="singlemodel"></a>

``` python
train(X_train, Y_train, Learning_rate, Epochs, Batch_size)
```

(Array) ```X_train``` is the data on which the network will be trained on

(Array) ```y_train``` are the labels corresponding to the data.

(Float) ```learning_rate``` if none then default set to 0.01.

(Integer) ```epochs``` are iterations of training the network goes through.

(Integer) ```batch_size``` is the batch that is being used from the whole training pool for each of the ```epochs```.

-------------------------------------------------------------------------------------

#### Evaluate (With trained network) - returns the accuracy of the network <a name="singlemodel"></a>

``` python
evaluate(X_test, Y_test)
```

(Array) ```X_test``` is the data that is given the Network for test purposes.

(Array) ```y_test``` are the labels, which need to be one-hot encoded.

-------------------------------------------------------------------------------------

#### Saving the trained model <a name="singlemodel"></a>

``` python
savem(Path)
```

(String) ```path``` defines the path where the pickle file containing the model data should be saved.

-------------------------------------------------------------------------------------

#### Loading in a saved model <a name="singlemodel"></a>

``` python
loadm(Path)
```

(String) ```path``` is the place where the pickle model file is stored.

Extra tips:

Instead of defining ```your_model_name = Model()``` you need to define ``` your_model_name = Model.loadm(your_path)```

-------------------------------------------------------------------------------------

#### Augmentation (to increase dataset size) - only for image datasets <a name="auxiliary"></a>

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

#### Image preprocessing (to convert images in arrays) - only for image datasets <a name="auxiliary"></a>

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

#### Tokenizer - only for strings <a name="auxiliary"></a>

``` python
tokenizer(Input_string)
```

(String) ```Input_string``` is the list which should be tokenized and vectorized according to the word array's indicies.

-------------------------------------------------------------------------------------

#### Normalization - for image data <a name="auxiliary"></a>

``` python
norm(X)
```

(Array) ```X``` is the data that should be normalized from 0-255 to 0-1.

-------------------------------------------------------------------------------------

#### Model Converter - Keras -> Singularity <a name="cross"></a>

```python
keras_to_Singularity(Keras_model)
```

(String ) ```keras_model``` is the keras model that is loaded in by the keras library itself.

-------------------------------------------------------------------------------------
  
#### Model Converter - Pytorch -> Singularity <a name="cross"></a>

```python
pytorch_to_Singularity(Pytorch_model)
```

(String ) ```pytorch_model``` is the pytorch model that is loaded in by the pytorch library itself.

-------------------------------------------------------------------------------------

#### predict - Multimodel <a name="multimodel"></a>

``` python
predict_multi(X):
```

(array) ```X``` is the array that should be predicted by the MultiModel
