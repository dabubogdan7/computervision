import cv2
import numpy as np
import pandas as pd
import threading
import pytesseract
import torch
import speech_recognition as sr
import json
import tkinter as tk
import mss
import pyautogui
import easyocr
import pyttsx3
import schedule
import requests
import openai
import pywhatkit
import dash
import plotly.express as px
import spacy
import fastapi
import uvicorn
import transformers
import langchain
import stable_baselines3
from flask import Flask, Response, jsonify, request
from deepface import DeepFace
from ultralytics import YOLO
from playsound import playsound
from fastapi import FastAPI
from huggingface_hub import hf_hub_download
from transformers import pipeline
from flask_mail import Mail, Message
from dash import dcc, html
import time
import datetime
import random
import matplotlib.pyplot as plt

# Configurare AI
model = YOLO("yolov8n.pt")  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Configurare Flask API
app = Flask(__name__)
@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_faces', methods=['POST'])
def detect_faces_api():
    file = request.files['image']
    image = np.array(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    faces = detect_faces(image)
    return jsonify({"faces_detected": len(faces)})

# Configurare Flask-Mail pentru trimiterea alertelor prin email
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_password'
mail = Mail(app)

def send_email_alert(subject, body):
    msg = Message(subject, sender='your_email@gmail.com', recipients=['recipient_email@gmail.com'])
    msg.body = body
    mail.send(msg)

# Interfață Tkinter complexă
class SurveillanceApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Surveillance System")
        self.root.geometry("1600x1000")
        self.surveillance_running = False
        self.screen_cap = mss.mss()
        self.audio_engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.speech_active = False

        tk.Label(self.root, text="🎥 AI Surveillance System", font=("Arial", 24, "bold"), fg="white", bg="black").pack(fill=tk.X)
        self.start_button = tk.Button(self.root, text="📷 Start Monitorizare", font=("Arial", 16), command=self.start_surveillance)
        self.start_button.pack(pady=10)
        self.stop_button = tk.Button(self.root, text="🛑 Stop Monitorizare", font=("Arial", 16), command=self.stop_surveillance)
        self.stop_button.pack(pady=10)

        self.log_text = tk.Text(self.root, height=20, width=120, bg="black", fg="green", font=("Courier", 12))
        self.log_text.pack(pady=10)
        self.update_log("Sistemul este gata...")

    def update_log(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)

    def start_surveillance(self):
        self.surveillance_running = True
        threading.Thread(target=self.monitor_screen).start()
        threading.Thread(target=self.monitor_audio).start()
        threading.Thread(target=self.check_suspicious_activity).start()
        threading.Thread(target=self.speech_recognition).start()
        self.update_log("Monitorizare pornită...")

    def stop_surveillance(self):
        self.surveillance_running = False
        self.update_log("Monitorizare oprită.")

    def monitor_screen(self):
        while self.surveillance_running:
            frame = self.capture_screen()
            faces = self.detect_faces(frame)
            objects = self.detect_objects(frame)
            text_detected = self.detect_text(frame)

            if faces or objects or text_detected:
                self.send_alert()

            cv2.imshow("Monitorizare AI", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def capture_screen(self):
        screenshot = self.screen_cap.grab(self.screen_cap.monitors[1])
        return np.array(screenshot)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return faces

    def detect_objects(self, frame):
        results = model(frame)
        return results

    def detect_text(self, frame):
        text = pytesseract.image_to_string(frame)
        if text:
            self.update_log(f"📖 Text Detectat: {text}")
            return True
        return False

    def send_alert(self):
        self.update_log("📧 Trimitere Alertă prin Email & WhatsApp")
        send_email_alert("Alertă Securitate", "Activitate suspectă detectată în sistemul AI.")
        pywhatkit.sendwhatmsg("+407XXXXXXXX", "Alerta! Activitate suspectă detectată!", 23, 59)
        playsound("alert.mp3")

    def check_suspicious_activity(self):
        while self.surveillance_running:
            self.update_log("🔍 Scanare activități suspecte...")
            time.sleep(10)

    def speech_recognition(self):
        with sr.Microphone() as source:
            while self.surveillance_running:
                self.update_log("🎤 Ascult...")
                try:
                    audio = self.recognizer.listen(source)
                    text = self.recognizer.recognize_google(audio)
                    self.update_log(f"🎙️ Text Detectat: {text}")
                    if "stop" in text.lower():
                        self.stop_surveillance()
                except sr.UnknownValueError:
                    self.update_log("🔇 Nu am înțeles, încerc din nou...")

    def run(self):
        self.root.mainloop()

# Rulează aplicația
if __name__ == "__main__":
    threading.Thread(target=app.run, kwargs={'port': 5000}).start()
    ai_app = SurveillanceApp()
    ai_app.run()