#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 01:02:25 2023

@author: Anil, Nandika, Rahul
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load data into a pandas DataFrame
data = pd.read_csv('combined_data.csv')
data = data.dropna()

# Separate the feature (textual data) and target (binary labels) columns
X = data['text']
y = data['isHate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the textual data using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

smote = SMOTE(random_state=42)
X_train_tfidf_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)


# Define and train the machine learning models
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf_resampled, y_train_resampled)

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf_resampled, y_train_resampled)


svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_tfidf_resampled, y_train_resampled)

# Simple Neural Network architecture
nn_model = Sequential()
nn_model.add(Dense(64, input_shape=(X_train_tfidf_resampled.shape[1],), activation='relu'))
nn_model.add(Dropout(0.5)) 
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dropout(0.5)) 
nn_model.add(Dense(1, activation='sigmoid'))
# Compile the model
nn_model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])


# Train the model
es = EarlyStopping(patience=50, restore_best_weights=True, monitor="loss")
nn_model.fit(X_train_tfidf_resampled.toarray(), y_train_resampled, epochs=10, batch_size=32, verbose=1, callbacks=[es])

# Make predictions on the testing set
lr_pred = lr_model.predict(X_test_tfidf)
nb_pred = nb_model.predict(X_test_tfidf)
svm_pred = svm_model.predict(X_test_tfidf)
nn_pred = nn_model.predict_classes(X_test_tfidf)


# Print the classification report and accuracy score for each model
print('Logistic Regression:')
print(classification_report(y_test, lr_pred))
print('Accuracy score:', accuracy_score(y_test, lr_pred))
print()


print('Naive Bayes:')
print(classification_report(y_test, nb_pred))
print('Accuracy score:', accuracy_score(y_test, nb_pred))
print()


print('Support Vector Machine:')
print(classification_report(y_test, svm_pred))
print('Accuracy score:', accuracy_score(y_test, svm_pred))
print()

print('Neural Network:')
print(classification_report(y_test, nn_pred))
print('Accuracy score:', accuracy_score(y_test, nn_pred))
