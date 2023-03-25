# Classify-movie-review-ML
we start by loading the IMDB dataset, which consists of movie reviews and their associated labels (positive or negative sentiment). We limit the number of words in our vocabulary to 10,000, and pad the sequences to a fixed length of 200.

We then define the deep learning architecture for our NLP task using the Sequential class from Keras. This includes an embedding layer, dropout layer, convolutional layer, max pooling layer, LSTM layer, and dense layer with a sigmoid activation function.

We compile the model using the compile method, which specifies the optimizer, loss function, and metrics to track during training. We then train the model using the fit method, which takes in the training data and labels, as well as the number of epochs and batch size.

After training, we evaluate the accuracy of the model on the test data using the evaluate method. We then make predictions on new data by creating new sequences of text using the texts_to_sequences method, padding them to the same length as the training data, and passing them through the model's predict method. Finally, we print out the predicted sentiment for each new review.
