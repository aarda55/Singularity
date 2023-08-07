# Documentation
This is the official Documentation of the Singularity Machine learning framework

-------------------------------------------------------------------------------------

# First/Input layer (Universal Input layer)

  ``` python
  Input_layer(Input_data, Input_shape, Out_shape)
  ```
  
  (Array/List)   ```Input_data``` is the training/validation data that is being passed. From there the input shape is also derived. Every array/list should have the        same shape.

  (Array) ```Input_shape``` is the shape the data has.

  (Array) ```Out_shape``` is the shape the output should be when forward propagation was passed.

# Augmentation (to increase dataset size) - only for image datasets

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

  (List) ```CATEGORIES``` is a list with the names of the folders containing the images.

  (Boolean) ```FLIP```, when activated flips every image and saves it in the same directory.

  (Boolean) ```NOISEE```, when activated adds Noise to every image and saves it in the same directory.

  (Boolean) ```ROTATE```, when activated rotates every image and saves it in the same directory.

# Image preprocessing (to convert images in arrays) - only for image datasets

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

  (List) ```CATEGORIES``` is a list with the names of the folders containing the images.

  (Integer) ```IMG_SIZE``` defines the size which all images are automatically reshaped.
