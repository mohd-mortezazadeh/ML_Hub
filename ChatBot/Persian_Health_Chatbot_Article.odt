
### چت‌بات بهداشتی گفت‌وگوکننده با استفاده از یادگیری عمیق

در این مقاله قصد داریم نحوه ساخت یک **چت‌بات بهداشتی گفت‌وگوکننده** را با استفاده از یادگیری عمیق بررسی کنیم. برای انجام این پروژه، نیاز است که با مفاهیم پایه یادگیری عمیق، مهارت‌های متوسط در زبان برنامه‌نویسی پایتون، و نظریه شبکه‌های عصبی آشنا باشید. همچنین، کاربران باید نحوه استفاده و پیکربندی ماژول **SpeechRecognition** را بدانند.

---

#### ۱. وارد کردن ماژول‌های مورد نیاز

ابتدا، ماژول‌های زیر را برای شروع فرآیند آموزش وارد می‌کنیم:

```python
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import numpy as np
```

**توضیح مختصر ماژول‌ها:**
- **nltk:** کتابخانه پردازش زبان طبیعی (NLP) در پایتون.
- **WordNetLemmatizer:** برای لِماتیزه کردن کلمات (حذف تغییرات صرفی و بازگرداندن کلمه به شکل پایه).
- **Sequential:** مدل ساده شبکه عصبی برای استفاده در چت‌بات.
- **Dense، Activation و Dropout:** لایه‌های شبکه عصبی؛ *Dense* لایه کاملاً متصل، *Dropout* برای جلوگیری از بیش‌برازش، و *Activation* برای فعال‌سازی نورون‌ها.
- **SGD:** روش گرادیان نزولی تصادفی برای بهینه‌سازی مدل.

---

#### ۲. ساخت داده‌ها و ساختار JSON

برای داده‌های آموزشی از یک فایل JSON با ساختاری مشابه زیر استفاده می‌کنیم:

```json
{
  "intents": [
    {
      "tag": "نام بیماری",
      "patterns": ["علائم جداشده با کاما"],
      "responses": ["پاسخی که کاربر دریافت می‌کند"]
    },
    ...
  ]
}
```

- **tag:** نام بیماری.
- **patterns:** مجموعه‌ای از علائم بیماری.
- **responses:** پاسخ‌هایی که به کاربر نمایش داده می‌شود.

---

#### ۳. پردازش داده‌ها

اطلاعات JSON را به سه متغیر اصلی تقسیم می‌کنیم:
- **words:** برای ذخیره علائم.
- **classes:** برای ذخیره نام بیماری‌ها.
- **documents:** ترکیبی از علائم و دسته‌بندی مربوط به آن‌ها.

کد زیر داده‌ها را پردازش کرده و کلمات را لِماتیزه می‌کند:

```python
lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))

        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))
```

---

#### ۴. ذخیره داده‌ها با استفاده از **pickle**

برای استفاده‌های بعدی، کلمات و دسته‌بندی‌ها را در قالب فایل **pickle** ذخیره می‌کنیم:

```python
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
```

---

#### ۵. آماده‌سازی داده‌ها برای آموزش مدل

برای ورودی‌های شبکه عصبی، داده‌ها را به مقادیر عددی تبدیل می‌کنیم. از روش **Bag of Words** برای نمایش داده‌ها استفاده می‌شود:

```python
dataset = []
template = [0]*len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(template)
    output_row[classes.index(document[1])] = 1
    dataset.append([bag, output_row])

random.shuffle(dataset)
dataset = np.array(dataset)

train_x = list(dataset[:, 0])
train_y = list(dataset[:, 1])
```

---

#### ۶. ساخت و آموزش مدل

مدل با استفاده از کتابخانه Keras ساخته و آموزش داده می‌شود:

```python
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save("chatbot_model.h5")
```

---

#### ۷. آزمایش چت‌بات

کد زیر یک چت‌بات کامل است که با استفاده از ورودی صوتی، علائم کاربر را شناسایی کرده و پاسخ مناسب را ارائه می‌دهد:

```python
import speech_recognition as sr
import pyttsx3

def calling_the_bot(txt):
    predict = predict_class(txt)
    res = get_response(predict, intents)

    engine.say(f"Based on the symptoms, the diagnosis is: {res}")
    engine.runAndWait()
    print(f"Symptoms: {txt}
Diagnosis: {res}")

if __name__ == "__main__":
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    engine = pyttsx3.init()

    print("Bot is running...")
    while True:
        with mic as source:
            print("Say your symptoms...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")
                calling_the_bot(text)
            except sr.UnknownValueError:
                print("Could not understand. Please try again.")
```

---

#### نتیجه‌گیری

این چت‌بات بهداشتی به صورت تعاملی علائم کاربر را دریافت کرده و بیماری محتمل را شناسایی می‌کند. می‌توانید این مدل را با داده‌های بیشتر گسترش داده و برای تشخیص بهتر استفاده کنید.
