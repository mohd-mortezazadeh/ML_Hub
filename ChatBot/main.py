import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import numpy as np
import speech_recognition as sr
import pyttsx3
import time

# Initialize lemmatizer for word normalization
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
intents = json.loads(open("intents.json").read())

# Load processed words and classes from pickle files
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the trained model
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    """Tokenizes and lemmatizes the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Creates a bag of words representation of the input sentence."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predicts the class of the input sentence using the trained model."""
    bow = bag_of_words(sentence)  # Convert sentence to bag of words
    res = model.predict(np.array([bow]))[0]  # Get predictions from the model

    ERROR_THRESHOLD = 0.25  # Set a threshold for confidence

    # Filter out predictions below the threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)  # Sort by probability

    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    """Fetches a random response for the predicted intent."""
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    result = ''
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])  # Get a random response
            break
    return result

def calling_the_bot(txt):
    """Processes the input text, predicts the intent, and responds via voice."""
    global res
    predict = predict_class(txt)  # Predict the class of the input
    res = get_response(predict, intents)  # Get the corresponding response

    # Speak the response using text-to-speech
    engine.say("Found it. From our database we found that " + res)
    engine.runAndWait()
    print("Your Symptom was: ", txt)
    print("Result found in our Database: ", res)

if __name__ == '__main__':
    print("Bot is Running")

    recognizer = sr.Recognizer()  # Initialize speech recognizer
    mic = sr.Microphone(device_index=4)  # Initialize microphone

    engine = pyttsx3.init()  # Initialize text-to-speech engine
    rate = engine.getProperty('rate')
    engine.setProperty('rate', 175)  # Set speech rate
    engine.setProperty('volume', 1.0)  # Set volume level

    voices = engine.getProperty('voices')

    # Greet the user
    engine.say("Hello user, I am Bagley, your personal Talking Healthcare Chatbot.")
    engine.runAndWait()

    # Ask for voice preference
    engine.say("IF YOU WANT TO CONTINUE WITH MALE VOICE PLEASE SAY MALE. OTHERWISE SAY FEMALE.")
    engine.runAndWait()

    # Capture voice input for gender preference
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.2)
        audio = recognizer.listen(source)

    audio = recognizer.recognize_google(audio)

    # Set voice based on user preference
    if audio.lower() == "female":
        engine.setProperty('voice', voices[1].id)  # Set female voice
        print("You have chosen to continue with Female Voice")
    else:
        engine.setProperty('voice', voices[0].id)  # Set male voice
        print("You have chosen to continue with Male Voice")

    # Main loop for symptom input
    while True:
        with mic as symptom:
            print("Say Your Symptoms. The Bot is Listening")
            engine.say("You may tell me your symptoms now. I am listening")
            engine.runAndWait()
            try:
                recognizer.adjust_for_ambient_noise(symptom, duration=0.2)
                symp = recognizer.listen(symptom)
                text = recognizer.recognize_google(symp)  # Convert audio to text
                engine.say("You said {}".format(text))
                engine.runAndWait()

                engine.say("Scanning our database for your symptom. Please wait.")
                engine.runAndWait()
                time.sleep(1)

                # Process the recognized symptoms
                calling_the_bot(text)
            except sr.UnknownValueError:
                # Handle unrecognized speech
                engine.say("Sorry, either your symptom is unclear to me or it is not present in our database. Please try again.")
                engine.runAndWait()
                print("Sorry, either your symptom is unclear to me or it is not present in our database. Please try again.")
            finally:
                engine.say("If you want to continue please say True otherwise say False.")
                engine.runAndWait()

        # Check if user wants to exit
        with mic as ans:
            recognizer.adjust_for_ambient_noise(ans, duration=0.2)
            voice = recognizer.listen(ans)
            final = recognizer.recognize_google(voice)

        if final.lower() == 'no' or final.lower() == 'please exit':
            engine.say("Thank You. Shutting Down now.")
            engine.runAndWait()
            break  # Exit the loop




"""
توضیحات کلی:
این کد یک چت‌بات صوتی به نام "Bagley" است که با استفاده از ورودی صوتی کاربر، علائم بیماری را شناسایی کرده و پاسخ‌های مربوطه را ارائه می‌دهد.
کد از کتابخانه‌های مختلفی مانند speech_recognition برای شناسایی گفتار، pyttsx3 برای تبدیل متن به گفتار و tensorflow برای بارگذاری مدل یادگیری عمیق استفاده می‌کند.
کاربر می‌تواند صدای مردانه یا زنانه را انتخاب کند و پس از آن می‌تواند علائم خود را بیان کند. چت‌بات پاسخ‌های مربوطه را از پایگاه داده خود پیدا کرده و به کاربر اعلام می‌کند.
"""