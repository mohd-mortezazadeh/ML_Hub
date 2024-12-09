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
try:
    intents = json.loads(open("intents.json").read())
except FileNotFoundError:
    print("Error: intents.json file not found.")
    exit(1)

# Load processed words and classes from pickle files
try:
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure 'words.pkl' and 'classes.pkl' exist.")
    exit(1)

# Load the trained model
try:
    model = load_model('chatbot_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

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
    if not intents_list:
        return "I'm sorry, I didn't understand that."
    
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
    engine.setProperty('rate', 140)  # Set speech rate (lower value for slower speech)
    engine.setProperty('volume', 1.0)  # Set volume level

    voices = engine.getProperty('voices')

    # Greet the user
    engine.say("Hello user, I am Siyamak, your personal Talking Healthcare Chatbot.")
    engine.runAndWait()

    # Ask for voice preference
    engine.say("IF YOU WANT TO CONTINUE WITH MALE VOICE PLEASE SAY MALE. OTHERWISE SAY FEMALE.")
    engine.runAndWait()

    # Capture voice input for gender preference
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source)
            audio = recognizer.recognize_vosk(audio)  # Use Vosk for recognition

            # Set voice based on user preference
            if audio.lower() == "female":
                engine.setProperty('voice', voices[1].id)  # Set female voice
                print("You have chosen to continue with Female Voice")
            else:
                engine.setProperty('voice', voices[0].id)  # Set male voice
                print("You have chosen to continue with Male Voice")

            # Main loop for symptom input
            while True:
                print("Say Your Symptoms. The Bot is Listening")
                engine.say("You may tell me your symptoms now. I am listening")
                engine.runAndWait()
                
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    symp = recognizer.listen(source)  # Use the same source
                    text = recognizer.recognize_vosk(symp)  # Convert audio to text
                    print(text, 11111111111111111111)
                    engine.say("You said {}".format(text))
                    engine.runAndWait()
                    time.sleep(2)

                    engine.say("Scanning our database for your symptom. Please wait.")
                    engine.runAndWait()
                    time.sleep(2)

                    # Process the recognized symptoms
                    calling_the_bot(text)
                except sr.UnknownValueError:
                    # Handle unrecognized speech
                    engine.say("Sorry, I could not understand what you said. Please try again.")
                    engine.runAndWait()
                    print("Sorry, I could not understand what you said. Please try again.")
                except sr.RequestError as e:
                    # Handle request error
                    engine.say("Could not request results from Google Speech Recognition service; {0}".format(e))
                    engine.runAndWait()
                    print(f"Could not request results from Google Speech Recognition service; {e}")

                # Check if user wants to exit
                engine.say("If you want to continue please say True otherwise say False.")
                engine.runAndWait()
                
                try:
                    voice = recognizer.listen(source)  # Use the same source
                    final = recognizer.recognize_vosk(voice)
                    if final.lower() == 'no' or final.lower() == 'please exit':
                        engine.say("Thank You. Shutting Down now.")
                        engine.runAndWait()
                        break  # Exit the loop
                except sr.UnknownValueError:
                    engine.say("Sorry, I did not understand that. Please try again.")
                    engine.runAndWait()
                except sr.RequestError as e:
                    engine.say("Could not request results from Google Speech Recognition service; {0}".format(e))
                    engine.runAndWait()
                    print(f"Could not request results from Google Speech Recognition service; {e}")

        except Exception as e:
            print(f"Error occurred: {e}")