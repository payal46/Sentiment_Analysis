# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#  Load our data
train_data = pd.read_csv("/content/train (1).csv")
test_data = pd.read_csv("/content/test (1).csv")

# Preprocessing of the data
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    return text

train_data['text'] = train_data['text'].apply(preprocess_text)
test_data['text'] = test_data['text'].apply(preprocess_text)

# Encode target variable
label_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
train_data['sentiment'] = train_data['sentiment'].map(label_mapping)

from imblearn.over_sampling import RandomOverSampler

#  Handle Class Imbalance because the classes are imbalanced in nature
X_raw = train_data['text']
y_raw = train_data['sentiment']
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_raw.values.reshape(-1, 1), y_raw)
X_resampled = X_resampled.flatten()

#  Tokenization
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_resampled)

X = tokenizer.texts_to_sequences(X_resampled)
X = pad_sequences(X, maxlen=150, padding='post', truncating='post')
y = y_resampled

X_test = tokenizer.texts_to_sequences(test_data['text'])
X_test = pad_sequences(X_test, maxlen=150, padding='post', truncating='post')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

import os

# Upload the Glove
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip

#  Load Pretrained Embeddings GloVe-
def load_embeddings(file_path, word_index, embedding_dim):
    embeddings_index = {}
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

embedding_dim = 100
embedding_matrix = load_embeddings("glove.6B.100d.txt", tokenizer.word_index, embedding_dim)

# Define the Model and their parameters
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=150,
                    trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Bidirectional(LSTM(128, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

# Training our model and I also use the earlystopping to take care of the Overfitting.
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    callbacks=[early_stopping]
)

# Evaluation of the model with different metrices.
val_preds = np.argmax(model.predict(X_val), axis=1)
accuracy = accuracy_score(y_val, val_preds)
precision = precision_score(y_val, val_preds, average='weighted')
recall = recall_score(y_val, val_preds, average='weighted')
f1 = f1_score(y_val, val_preds, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Final Task - Predictions on Test Data with our model
test_preds = np.argmax(model.predict(X_test), axis=1)

reverse_label_mapping = {v: k for k, v in label_mapping.items()}
test_data['sentiment'] = [reverse_label_mapping[pred] for pred in test_preds]

# Save predictions with the file name- Test_predictions"
test_data.to_csv("test_predictions.csv", index=False)




















