import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
import datetime
from deepface import DeepFace
import mediapipe as mp
import google.generativeai as genai
import json
import speech_recognition as sr
import threading

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Global variables for behavior analysis
behavior_report = []
emotion_totals = {"happy": 0, "sad": 0, "angry": 0, "surprise": 0, "fear": 0, "disgust": 0, "neutral": 0}
frame_count = 0
hand_gesture_count = {"tense": 0, "relaxed": 0}
eye_direction_count = {"left": 0, "right": 0, "center": 0}
head_movement_count = {"up": 0, "down": 0, "still": 0}
session_log = ""  # To store speech-to-text session
stop_speech_thread = False  # Flag to stop the speech-to-text thread
gemini_qa = []  # To store questions and answers from Gemini


# Function to log data to the report
def log_to_report(mode, analysis):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    behavior_report.append(f"{timestamp} - {mode}: {analysis}")


# Function to analyze emotion using DeepFace
def analyze_emotion(frame):
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    emotion_scores = result[0]['emotion']
    return emotion_scores


# Function to analyze hand gestures
def analyze_hands(frame, hands):
    global hand_gesture_count
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Example: Check if the hand is tense (fingers closed) or relaxed (fingers open)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y)

            if distance < 0.1:
                hand_gesture_count["tense"] += 1
            else:
                hand_gesture_count["relaxed"] += 1

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


# Function to analyze eye direction and head movement
def analyze_face(frame, face_mesh):
    global eye_direction_count, head_movement_count
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Example: Check eye direction (left, right, center)
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]

            if left_eye.x < 0.4:
                eye_direction_count["left"] += 1
            elif right_eye.x > 0.6:
                eye_direction_count["right"] += 1
            else:
                eye_direction_count["center"] += 1

            # Example: Check head movement (up, down, still)
            nose_tip = face_landmarks.landmark[4]  # Nose tip landmark
            if nose_tip.y < 0.4:
                head_movement_count["up"] += 1
            elif nose_tip.y > 0.6:
                head_movement_count["down"] += 1
            else:
                head_movement_count["still"] += 1

            # Draw face landmarks on the frame
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)


# Function to handle Speech-to-Text in a separate thread
def speech_to_text():
    global session_log, stop_speech_thread
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("🎤 ابدأ التحدث... (قل 'توقف' أو 'Stop' للخروج)")
        recognizer.adjust_for_ambient_noise(source)

        while not stop_speech_thread:
            try:
                print("🎙️ استماع...")
                audio = recognizer.listen(source, phrase_time_limit=10)
                text = recognizer.recognize_google(audio, language="ar-EG,en-US")

                print(f"📢 النص المسموع: {text}")

                if "توقف" in text or "stop" in text.lower():
                    print("\n⏹️ تم إنهاء الجلسة.")
                    stop_speech_thread = True
                    break

                session_log += text + "\n"

            except sr.UnknownValueError:
                print("❌ لم يتم التعرف على الكلام، حاول مرة أخرى...")
            except sr.RequestError:
                print("⚠️ خطأ في الاتصال بخدمة التعرف على الصوت!")


# Function to send session log to Gemini
def send_session_to_gemini(session_log, mode):
    """Send the session log to Gemini for analysis."""
    prompt = f"""
    لديك هذا الحوار الصوتي، استخرج الأسئلة والإجابات وحدد هل كل إجابة صحيحة أم خاطئة.
    المود المستخدم: {mode}

    الحوار:
    {session_log}

    أعد الإجابة بصيغة JSON كالتالي:
    {{
        "qa_list": [
            {{
                "question": "السؤال المستخرج",
                "answer": "الإجابة المستخرجة",
                "is_correct": true/false,
                "correct_answer": "الإجابة الصحيحة إن وجدت"
            }},
            ...
        ]
    }}
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    # Parse the JSON response
    try:
        qa_list = json.loads(response.text)["qa_list"]
        return qa_list
    except Exception as e:
        print(f"⚠️ Error parsing Gemini response: {e}")
        return []


