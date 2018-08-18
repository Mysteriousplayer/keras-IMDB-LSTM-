from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
vocabulary_size = 5000
max_words = 500
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = 
vocabulary_size)
print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train), len(X_test)))
print('---review---')
print(X_train[6])
print('---label---')
print(y_train[6])

print('Maximumreview length: {}'.format(len(max((X_train+ X_test), key=len))))

X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

embedding_size=32
model=Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100,dropout=0.2,return_sequences=True))#return_sequences=True！
#model.add(activation('sigmoid'))
model.add(LSTM(100,dropout=0.2))#多一层神经网络
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

batch_size = 64
num_epochs = 3
X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
scores =model.evaluate(X_test, y_test, verbose=0)
print('Testaccuracy:', scores[1])