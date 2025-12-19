import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# -------------------------------
# Training Data
# -------------------------------
sentences = [
    "I love my dog",
    "I love my cat",
    "You love my dog",
    "I love a dog",
    "Do you think my dog is amazing",
    "I love machine learning",
    "Machine learning is amazing"
]

# -------------------------------
# Tokenization
# -------------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1

# Create n-gram sequences
input_sequences = []
for sentence in sentences:
    token_list = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])

max_sequence_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

# -------------------------------
# Model
# -------------------------------
model = Sequential([
    Embedding(total_words, 100),
    SimpleRNN(150),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=0)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🧠 NLP Next Word Prediction App")
st.write("Enter a sentence and predict the **next word**")

user_input = st.text_input("Enter text")

if st.button("Predict Next Word"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        token_list = tokenizer.texts_to_sequences([user_input])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        prediction = model.predict(token_list)
        predicted_index = np.argmax(prediction)

        predicted_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                predicted_word = word
                break

        st.success(f"✅ Predicted Next Word: **{predicted_word}**")
