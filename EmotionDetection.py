import tkinter as tk
from tkinter import filedialog
from tkinter import *
from sklearn import metrics
from tensorflow.keras.models import model_from_json
from keras.utils import register_keras_serializable
from keras.engine.functional import Functional
from PIL import Image, ImageTk
import numpy as np
import cv2

# Register the 'Functional' class
register_keras_serializable()(Functional)

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

top = tk.Tk()
top.geometry('850x650')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

title_label = Label(top, text="Emotion Detector", background='#CDCDCD', font=('arial', 40, 'bold'))
title_label.pack(pady=20)

label1 = Label(top, background='#CDCDCD', font=('arial', 25, 'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model_a2.json", "model_weights.weights.h5")

EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOTIONS_COLORS = ["#FF0000", "#FF4500", "#8A2BE2", "#FFD700", "#808080", "#4682B4", "#32CD32"]

def Detect(file_path):
    global label1
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image, 1.3, 5)
    try:
        for (x, y, w, h) in faces:
            fc = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
        print("Predicted Emotion is " + pred)
        color = EMOTIONS_COLORS[EMOTIONS_LIST.index(pred)]
        label1.configure(foreground=color, text=pred)
    except:
        label1.configure(foreground="#011638", text="Unable to detect")

def show_Detect_button(file_path):
    detect_b = Button(top, text="Predict Emotion", command=lambda: Detect(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 15, 'bold'))
    detect_b.place(relx=0.6, rely=0.6)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except:
        pass

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 15, 'bold'))
upload.place(relx=0.4, rely=0.6)

sign_image.pack(side=TOP, expand=True, pady=20)
label1.pack(side=TOP, expand=True, pady=20)

# Add emotion images
emotion_frame = Frame(top, background='#fff')
emotion_frame.pack(side=TOP, pady=10)

for emotion in EMOTIONS_LIST:
    img = Image.open(f'emotions/{emotion}.jpg')
    img.thumbnail((100, 100))
    imgtk = ImageTk.PhotoImage(img)
    panel = Label(emotion_frame, image=imgtk, background='#CDCDCD')
    panel.image = imgtk
    panel.pack(side=LEFT, padx=10)

top.mainloop()
