import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()


with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)



words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model("chatbot_model.h5")

def clean_up_sentence(sentence):
    """Tokenizes and lemmatizes the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Creates a bag-of-words array from the sentence."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predicts the intent of the sentence."""
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.01
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    """Gets the chatbot's response based on predicted intent."""
    if not intents_list:
        return random.choice([
            "Sorry, I didn’t understand that.",
            "Could you rephrase?",
            "I’m not sure I got that."
        ])
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

print("✅ Bot is running! Type 'quit' to exit.\n")

while True:
    message = input("You: ")
    if message.lower() == "quit":
        print("👋 Goodbye!")
        break
    if not message.strip():
        print("⚠️ Please type something.")
        continue
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(f"Bot: {res}")
