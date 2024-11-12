import random
import json
import pickle
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Download required NLTK resources
nltk.download('punkt')  # برای توکن‌سازی جملات
nltk.download('wordnet')  # برای لِمَت‌سازی کلمات
nltk.download('omw-1.4')  # برای دسترسی به WordNet

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
with open("intents.json") as file:
    intents = json.load(file)  # بارگذاری داده‌های intents از فایل JSON

words = []  # لیستی برای ذخیره کلمات
classes = []  # لیستی برای ذخیره کلاس‌ها (برچسب‌ها)
documents = []  # لیستی برای ذخیره مستندات (جملات و برچسب‌ها)

ignore_letters = ["?", "!", ".", ","]  # کاراکترهایی که باید نادیده گرفته شوند

# Process intents to create training data
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)  # توکن‌سازی جملات
        words.extend(word_list)  # اضافه کردن کلمات به لیست words
        documents.append((word_list, intent["tag"]))  # ذخیره جملات و برچسب‌ها

        if intent["tag"] not in classes:
            classes.append(intent["tag"])  # اضافه کردن برچسب به کلاس‌ها

# Save words and classes to pickle files
pickle.dump(words, open('words.pkl', 'wb'))  # ذخیره کلمات در فایل
pickle.dump(classes, open('classes.pkl', 'wb'))  # ذخیره کلاس‌ها در فایل

# Prepare dataset for training
dataset = []  # لیست برای ذخیره داده‌های آموزشی
template = [0] * len(classes)  # الگوی خروجی برای هر کلاس

# ایجاد داده‌های آموزشی
for document in documents:
    bag = []  # لیست برای ذخیره بردار ورودی
    word_patterns = document[0]  # دریافت کلمات الگو
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]  # لِمَت‌سازی کلمات

    # ایجاد بردار ورودی (bag of words)
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # ایجاد بردار خروجی
    output_row = list(template)
    output_row[classes.index(document[1])] = 1  # تنظیم مقدار 1 برای کلاس مربوطه
    dataset.append([bag, output_row])  # اضافه کردن داده به dataset

# Shuffle dataset to ensure randomness
random.shuffle(dataset)  # مخلوط کردن داده‌ها

# Check lengths before converting to numpy array
for entry in dataset:
    bag, output_row = entry
    if len(bag) != len(words) or len(output_row) != len(classes):
        print(f"Length mismatch: Bag length: {len(bag)}, Output row length: {len(output_row)}")

# Convert dataset to numpy array
try:
    dataset = np.array(dataset, dtype=object)  # استفاده از dtype=object برای آرایه‌های با طول متغیر
except ValueError as e:
    print("Error converting dataset to numpy array:", e)

# Split dataset into training data
train_x = np.array([item[0] for item in dataset])  # استخراج بردارهای ورودی
train_y = np.array([item[1] for item in dataset])  # استخراج بردارهای خروجی

# Build the neural network model
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))  # لایه ورودی
model.add(Dropout(0.5))  # لایه Dropout برای جلوگیری از Overfitting
model.add(Dense(128, activation='relu'))  # لایه مخفی
model.add(Dropout(0.5))  # لایه Dropout
model.add(Dense(len(train_y[0]), activation='softmax'))  # لایه خروجی با تابع فعال‌سازی softmax

# Compile the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)  # تنظیمات بهینه‌ساز
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  # کامپایل مدل

# Train the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)  # آموزش مدل

# Save the model
model.save("chatbot_model.h5")  # ذخیره مدل آموزش‌دیده
print("Done!")  # پایان کار


'''
توضیحات کلی:
این کد یک مدل یادگیری عمیق برای شناسایی گفتار و پاسخ به آن با استفاده از داده‌های ورودی از فایل JSON ایجاد می‌کند.
ابتدا داده‌ها پردازش می‌شوند و سپس مدل آموزش داده می‌شود.
در نهایت، مدل آموزش‌دیده در یک فایل H5 ذخیره می‌شود.

'''