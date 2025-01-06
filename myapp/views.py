import pickle
from django.shortcuts import render
from django.http import JsonResponse
from nltk.stem import WordNetLemmatizer
import nltk
import string
from textblob import TextBlob
import random
from keras.models import load_model # type: ignore
import json

# Load 
intents = json.loads(open('models/dataset.json').read())
model = load_model('models/model.h5')
words = pickle.load(open('models/words.pkl','rb'))
classes = pickle.load(open('models/class.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb')) 

lemmatizer = WordNetLemmatizer()

# Koteksi kata menggunakan TextBlob
def spell_correction(text):
    return str(TextBlob(text).correct())

# Pre processing
def preprocess_text(sentence):
    # Koreksi ejaan
    corrected_sentence = spell_correction(sentence)
    # Tokenisasi dan lemmatization
    tokens = nltk.word_tokenize(corrected_sentence)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in string.punctuation]
    return ' '.join(tokens)

# Predict
def predict_class(sentence, model):
    # Transform ke numerik menggunakan TF-IDF
    processed_sentence = preprocess_text(sentence)
    features = vectorizer.transform([processed_sentence]).toarray()

    # Predict 
    res = model.predict(features)[0]
    # Gunakan threshold agar probabilitas yang lebih besar dari 0.25 yang akan digunakan sebagai input yang valid
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Return intent dan probabilitasnya
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# Response
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

def chatbot_response(user_message):
    intents_list = predict_class(user_message, model)
    return get_response(intents_list, intents)

# Mengembalikan response dalam bentuk JSON
def get_bot_response(request):
    if request.method == 'GET':
        user_message = request.GET.get('msg', '')
        response = chatbot_response(user_message)
        return JsonResponse({'response': response})
    return JsonResponse({'error': 'Invalid request'}, status=400)

def index(request):
    return render(request, 'index.html')
