# Loading TensorFlow or Keras

# From https://keras.rstudio.com/

# Keras

library(keras)
mnist <- dataset_mnist()

# Extract the training and test data.

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Reshape 28x28 pixels greyscale.
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# Normalize greyscale to 0-1.  (Max-min normalization.)
x_train <- x_train / 255
x_test <- x_test / 255

# One-hot encoding of digit identification (0 if not the digit, 1 if the digit for 10 columns 0-9)
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

model <- keras_model_sequential() 
model %>% #256, 128
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

# Load optimization model and cost function.
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = rms_prop(),
#  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

# Training data with cross validation split of certain percentage.
# Does it resplit to validate on different epoch?
history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

# Run the model on the test data.
model %>% evaluate(x_test, y_test)

# Generate predictions on new data.
model %>% predict_classes(x_test)

# Would it be useful to indentify which predictions are correct and which are not?  If so, maybe some insight to why it does or doesn't work can be gained.

# Example Change - I can't remember what this next part of the code does.  Does it train and test on the same data?
y_new <- model %>% predict_classes(x_test)
y_new <- to_categorical(y_new, 10)

model %>% evaluate(x_test, y_new)
