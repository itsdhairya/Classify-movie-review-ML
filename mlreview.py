import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the IMDB dataset
num_words = 10000
maxlen = 200
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)

# Pad the sequences to a fixed length
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# Define the deep learning architecture
model = keras.Sequential()
model.add(layers.Embedding(num_words, 32, input_length=maxlen))
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(64, 5, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=4))
model.add(layers.LSTM(64))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.2f}')

# Make predictions on new data
new_reviews = ['This movie was terrible', 'I loved this movie']
new_sequences = keras.preprocessing.text.texts_to_sequences(new_reviews)
new_sequences = keras.preprocessing.sequence.pad_sequences(new_sequences, maxlen=maxlen)
predictions = model.predict(new_sequences)
for i in range(len(new_reviews)):
    print(f'{new_reviews[i]} - Positive' if predictions[i] > 0.5 else f'{new_reviews[i]} - Negative')
