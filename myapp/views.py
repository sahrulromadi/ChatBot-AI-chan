import pickle
from django.shortcuts import render
from django.http import JsonResponse
from nltk.stem import WordNetLemmatizer
import nltk
import string
from textblob import TextBlob
import random
from keras.models import load_model  # type: ignore
import json

# Load model dan data sekali di awal
with open('models/dataset.json') as file:
    intents = json.load(file)
    
model = load_model('models/model.h5')
words = pickle.load(open('models/words.pkl', 'rb'))
classes = pickle.load(open('models/class.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()

# Koreksi ejaan menggunakan TextBlob
def spell_correction(text):
    return str(TextBlob(text).correct())

# Cleaning dan preprocessing input teks
def cleaning_text(sentence):
    # Koreksi ejaan
    corrected_sentence = spell_correction(sentence)
    # Tokenisasi dan lemmatization
    tokens = nltk.word_tokenize(corrected_sentence)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in string.punctuation]
    return ' '.join(tokens)

# Prediksi intent menggunakan model tanpa threshold
def predict_class(sentence, model):
    # Proses input yang sudah di-cleaning
    processed_sentence = cleaning_text(sentence)
    # Transform ke numerik menggunakan TF-IDF
    features = vectorizer.transform([processed_sentence]).toarray()

    # Prediksi model untuk setiap intent
    res = model.predict(features)[0]

    # Ambil index dengan probabilitas tertinggi
    max_index = res.argmax()
    max_prob = res[max_index]

    # Return intent dengan probabilitas tertinggi
    return [{"intent": classes[max_index], "probability": str(max_prob)}]

# Mendapatkan response berdasarkan intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']: 
        # Pilih tag dengan probabilitas tertinggi
        if intent['tag'] == tag:
            return random.choice(intent['responses']) # Ambil response secara acak
    
# Fungsi utama untuk menghasilkan response chatbot
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
