import pickle
import sys
import random
import os
import tensorflow as tf
import numpy as np
import sklearn as sk
from operator import itemgetter
from collections import Counter
from tensorflow.keras.layers import TextVectorization

'''Text encoding'''
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')


'''File Text handling and parsing'''
# Source file retrieving
def get_file():
    with open('texts_source.txt', 'r') as source:
        return source.read().strip()

# Parse Texts stored in a .txt
def parse_texts():
    filename=get_file()
    texts = []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            # Rough check for timestamp
            try:
                int(line[0])
                person = line[(line.index('-') + 2):(line.index(':', line.index('-')))]
                text = line[(line.index(':', line.index('-')) + 1):(line.index('\n'))].strip()
                text = text.encode('ascii', errors='ignore').decode('ascii') # Remove emojis
                if text and person:
                    texts.append([person, text])
            except:
                continue

    with open('parsed_texts.pkl', 'wb') as f:
        pickle.dump(texts, f)
    print('Parsed and saved to parsed_texts.pkl')
    return texts

# Load Parsed Texts
def load_texts():
    with open('parsed_texts.pkl', 'rb') as f:
        return pickle.load(f)


'''Search keyword usage'''

def search_text(texts, keyword):
    count = 0
    freq = {}

    for person, msg in texts:
        if keyword.lower() in msg.lower():
            safe_text = f"{person}: {msg}".encode('ascii', errors='ignore').decode('ascii')
            print(safe_text)
            freq[person] = freq.get(person, 0) + 1
            count += 1

    print(f"\nWord/phrase '{keyword}' found {count} times")
    for person, count in sorted(freq.items(), key=itemgetter(1), reverse=True):
        print(f"{person}: {count} times")



''' Build & Train AI Model '''

def create_model(texts):
    # Filter out small reply messages such as 'ok' and users with low amount of texts
    filtered_texts = []

    for user, message in texts:
        if len(message.split()) >= 3:
            filtered_texts.append((user, message))

    counts = Counter([user for user, _ in filtered_texts])
    filtered_texts = [(user, message) for user, message in filtered_texts if counts[user] >= len(filtered_texts)*0.02]

    users = [pers for pers, _ in filtered_texts]
    messages = [msg for _, msg in filtered_texts]

    # Encode users for the model to understand
    encoder = sk.preprocessing.LabelEncoder()
    labels = encoder.fit_transform(users)

    # Train-test split with stratification
    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(
        messages,
        labels,
        test_size=0.3,
        stratify=labels,
        random_state=42
    )

    # Covert data to tensors for network
    x_train = tf.constant(x_train, dtype=tf.string)
    x_test = tf.constant(x_test, dtype=tf.string)
    y_train = tf.constant(y_train, dtype=tf.int32)
    y_test = tf.constant(y_test, dtype=tf.int32)

    # Class weighting implementation to reduce user bias due to message imbalance among users
    y_train_np = y_train.numpy()
    class_weights = sk.utils.class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train_np),
        y=y_train_np
    )
    class_weights = dict(enumerate(class_weights))

    # Text-Vectorizer implementation
    vectorizer = TextVectorization(
        max_tokens=10000, 
        output_mode='int', 
        output_sequence_length=150
        )
    vectorizer.adapt(x_train)

    # API model implementation
    # Input layer consisting of all messages
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
    # Text-Vectorizer
    x = vectorizer(inputs)
    # Embedding layer
    x = tf.keras.layers.Embedding(input_dim=10000, output_dim=128)(x)
    # Bidirectional LSTM layer for message interpretation
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(x)
    # Hidden dense layer
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    # Dropout
    x = tf.keras.layers.Dropout(0.3)(x)
    # Output layer consisting of all users
    outputs = tf.keras.layers.Dense(len(encoder.classes_), activation='softmax')(x)

    # Model creation, compilation and fitting using class weights and early stopping
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
        )
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

    model.fit(
        x_train, 
        y_train, 
        epochs=15, 
        validation_data=(x_test, y_test), 
        class_weight=class_weights,
        callbacks=[callback]
        )

    # Save model and encoder
    model.save('chat_user_classifier.keras')
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    print("Model trained & saved")
    return model, encoder


'''Predict with Distribution'''

def predict_distribution(model, encoder, message):
    # Based on the trained model, predicts and returns all probabilities of message author

    message_tensor = tf.constant([message], dtype=tf.string)
    probs = model.predict(message_tensor, verbose=2)[0]

    print("\nPrediction Distribution:")
    for author, p in zip(encoder.classes_, probs):
        print(f"{author}: {p:.4f}")

    # Most likely author
    predicted_author = encoder.inverse_transform([probs.argmax()])[0]
    print(f"\nFinal author: {predicted_author}")


'''Main'''

if __name__ == '__main__':
    # Transforming terminal word/phrase input into correct format
    if len(sys.argv) > 2:
        phrase = ''
        for index, word in enumerate(sys.argv[2:]):
            phrase += word + ' ' if index != len(sys.argv[2:]) - 1 else word

    # Predict/ phrase
    if sys.argv[1] == 'train':
        try:
            with open('texts_source.txt', 'w') as source:
                source.write(phrase) 
        except:
            raise ValueError(f"File {phrase} not found or incorrect file format, must be .txt")
        model, encoder = create_model(parse_texts())
    elif sys.argv[1] == 'predict':
        model = tf.keras.models.load_model(
            'chat_user_classifier.keras',
            custom_objects={"TextVectorization": TextVectorization}
            )
        with open('label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        predict_distribution(model, encoder, phrase)
    elif sys.argv[1] == 'search':
        # Search phrase
        search_text(load_texts(), phrase)
    else:
        raise ValueError("Usage: main.py (search/predict/train) (word/phrase to search or predict, or filename for training)")
   
# Program by Pedro Oubiña S. 2025