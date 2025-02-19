import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import datetime
from deepface import DeepFace
import pandas as pd
import random
from PIL import Image
from torchvision import transforms

def augment_images(dataset, augmentations=20):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
    ])

    augmented_dataset = []

    for img, label in dataset:
        augmented_dataset.append((img, label))

        for i in range(augmentations):
            augmented_img = transform(img)
            augmented_dataset.append((augmented_img, label))

    return augmented_dataset

# Example usage:
dataset = [(Image.open('C:/Users/kenzy/Downloads/nouran.jpg'), 'nouran'), (Image.open('C:/Users/kenzy/Downloads/nouran2.jpg'), 'nouran'), (Image.open('C:/Users/kenzy/Downloads/nouran3.jpg'), 'nouran'), (Image.open('C:/Users/kenzy/Downloads/nouran4.jpg'), 'nouran'), (Image.open('C:/Users/kenzy/Downloads/kenzy5.jpg'), 'kenzy') ,  (Image.open('C:/Users/kenzy/Downloads/kenzy4.jpg'), 'kenzy') ,  (Image.open('C:/Users/kenzy/Downloads/kenzy3.jpg'), 'kenzy') , (Image.open('C:/Users/kenzy/Downloads/kenzy2.jpg'), 'kenzy'), (Image.open('C:/Users/kenzy/Downloads/kenzy.jpg'), 'kenzy') , (Image.open('C:/Users/kenzy/Downloads/abd.jpg'), 'abdelrahman') , (Image.open('C:/Users/kenzy/Downloads/abd2.jpg'), 'abdelrahman') , (Image.open('C:/Users/kenzy/Downloads/abd3.jpg'), 'abdelrahman')  , (Image.open('C:/Users/kenzy/Downloads/abd4.jpg'), 'abdelrahman') , (Image.open('C:/Users/kenzy/Downloads/omar.jpg'), 'omar') ,  (Image.open('C:/Users/kenzy/Downloads/omar2.jpg'), 'omar') ,  (Image.open('C:/Users/kenzy/Downloads/omar3.jpg'), 'omar') ,  (Image.open('C:/Users/kenzy/Downloads/omar4.jpg'), 'omar')   , (Image.open('C:/Users/kenzy/Downloads/dr.jpg'), 'Dr Mahmoud Farag')   , (Image.open('C:/Users/kenzy/Downloads/dr2.jpg'), 'Dr Mahmoud Farag') , (Image.open('C:/Users/kenzy/Downloads/dr3.jpg'), 'Dr Mahmoud Farag') , (Image.open('C:/Users/kenzy/Downloads/dr4.jpg'), 'Dr Mahmoud Farag') , (Image.open('C:/Users/kenzy/Downloads/dr5.jpg'), 'Dr Mahmoud Farag')]

augmented_dataset = augment_images(dataset)

# Extract face embeddings from the augmented dataset
face_embeddings = []
for img, label in augmented_dataset:
    img_np = np.array(img)
    embedding_dict = DeepFace.represent(img_np, model_name='Facenet', enforce_detection=False)
    embedding = embedding_dict[0]['embedding']  # Extract the embedding list
    face_embeddings.append((embedding, label))

# A global variable to store the accumulated behavior data for the report
behavior_report = []
emotion_totals = {"happy": 0, "sad": 0, "angry": 0, "surprise": 0, "fear": 0, "disgust": 0, "neutral": 0}
frame_count = 0

def log_to_report(mode, analysis):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    behavior_report.append(f"{timestamp} - {mode}: {analysis}")

def analyze_emotion(frame):
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    emotion_scores = result[0]['emotion']
    return emotion_scores


import time

import time

def start_analysis(mode):
    global frame_count
    cap = cv2.VideoCapture(0)  # Use the camera

    start_time = time.time()
    name_frequency = {}
    dominant_label = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze the emotion using DeepFace
        emotion_scores = analyze_emotion(frame)

        # Accumulate the emotion percentages over time (for each frame)
        global emotion_totals
        for emotion, score in emotion_scores.items():
            emotion_totals[emotion] += score
        frame_count += 1

        # Find the dominant emotion
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        dominant_emotion_score = emotion_scores[dominant_emotion]

        # Display the dominant emotion and its percentage
        emotion_text = f"{dominant_emotion}: {dominant_emotion_score:.2f}%"
        frame = cv2.putText(frame, emotion_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        # Mode-based behavior analysis
        if mode == "Detective":
            log_to_report(mode, f"Detecting: {dominant_emotion} ({dominant_emotion_score:.2f}%)")
        elif mode == "Student Behavior":
            log_to_report(mode, f"Tracking: {dominant_emotion} ({dominant_emotion_score:.2f}%)")
        elif mode == "Interview":
            log_to_report(mode, f"Analyzing: {dominant_emotion} ({dominant_emotion_score:.2f}%)")

        # Detect faces in the frame
        try:
            faces = DeepFace.extract_faces(frame, enforce_detection=False)
            for face in faces:
                x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']

                # Only update the name frequency and dominant label for the first 2 seconds
                if time.time() - start_time < 2:
                    face_img = frame[y:y+h, x:x+w]
                    face_embedding_dict = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)
                    face_embedding = face_embedding_dict[0]['embedding']  # Extract the embedding list

                    # Compare the face embedding with the embeddings from the augmented dataset
                    min_distance = float('inf')
                    recognized_name = "Unknown"
                    for embedding, label in face_embeddings:
                        distance = np.linalg.norm(np.array(face_embedding) - np.array(embedding))
                        if distance < min_distance:
                            min_distance = distance
                            recognized_name = label

                    # Update the frequency of the recognized name
                    if recognized_name in name_frequency:
                        name_frequency[recognized_name] += 1
                    else:
                        name_frequency[recognized_name] = 1

                # After 2 seconds, fix the dominant label
                if time.time() - start_time >= 2 and dominant_label is None:
                    if name_frequency:
                        dominant_label = max(name_frequency, key=name_frequency.get)
                    else:
                        dominant_label = "Unknown"

                # Display the dominant label and rectangle on the frame
                label_to_display = dominant_label if dominant_label is not None else recognized_name
                cv2.putText(frame, label_to_display, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        except Exception as e:
            print(f"Error in face detection: {e}")

        # Display the frame with the predicted emotion and percentages
        cv2.imshow("Frame", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Generate the report based on accumulated emotions
    generate_report(mode)

def analyze_behavior(average_emotion_scores, mode):
    behavior = ""
    actions = ""

    if mode == "Detective":
        if average_emotion_scores["happy"] > 50:
            behavior = "The person appears calm and confident, possibly truthful."
            actions = "Continue the conversation without pressure."
        elif average_emotion_scores["angry"] > 50:
            behavior = "The person seems defensive or frustrated, possibly hiding something."
            actions = "Attempt to de-escalate the situation and build trust."
        elif average_emotion_scores["sad"] > 50:
            behavior = "The person is showing signs of sadness or remorse."
            actions = "Be empathetic and avoid aggressive questioning."
        elif average_emotion_scores["surprise"] > 50:
            behavior = "The person might be caught off guard or reacting to unexpected information."
            actions = "Probe the reaction to understand what surprised them."
        elif average_emotion_scores["fear"] > 50:
            behavior = "The person is likely anxious or fearful, possibly hiding something."
            actions = "Reassure them and create a safe space for dialogue."
        elif average_emotion_scores["disgust"] > 50:
            behavior = "The person might be uncomfortable or disgusted by the topic."
            actions = "Change the topic or tone down the intensity."
        elif average_emotion_scores["neutral"] > 50:
            behavior = "The person is emotionally distant, possibly hiding their feelings."
            actions = "Ask open-ended questions to encourage emotional engagement."

    elif mode == "Student Behavior":
        if average_emotion_scores["happy"] > 50:
            behavior = "The student seems engaged, motivated, and in a positive state."
            actions = "Encourage further participation and offer praise."
        elif average_emotion_scores["angry"] > 50:
            behavior = "The student may be frustrated or upset with the material."
            actions = "Offer a break or adjust the learning approach."
        elif average_emotion_scores["sad"] > 50:
            behavior = "The student is feeling down or overwhelmed."
            actions = "Provide emotional support and check for any personal issues."
        elif average_emotion_scores["fear"] > 50:
            behavior = "The student might be anxious or scared of failure."
            actions = "Provide reassurance and reduce the pressure."
        elif average_emotion_scores["disgust"] > 50:
            behavior = "The student seems disengaged or repelled by the content."
            actions = "Change the subject or provide a different approach."
        elif average_emotion_scores["neutral"] > 50:
            behavior = "The student is indifferent or not emotionally engaged."
            actions = "Try interactive methods or involve them in discussions."

    elif mode == "Interview":
        if average_emotion_scores["happy"] > 50:
            behavior = "The candidate seems confident and positive about the opportunity."
            actions = "Encourage them to elaborate on their achievements and enthusiasm."
        elif average_emotion_scores["angry"] > 50:
            behavior = "The candidate seems frustrated or defensive, possibly due to stress."
            actions = "Attempt to make the candidate feel comfortable and ask clarifying questions."
        elif average_emotion_scores["sad"] > 50:
            behavior = "The candidate may be feeling down or lack confidence."
            actions = "Provide reassurance and give them time to gather their thoughts."
        elif average_emotion_scores["fear"] > 50:
            behavior = "The candidate is likely anxious or fearful about the interview."
            actions = "Reassure them and reduce the pressure to help them relax."
        elif average_emotion_scores["disgust"] > 50:
            behavior = "The candidate might be uncomfortable with the questions or the situation."
            actions = "Adjust your tone and make the interview feel less intimidating."
        elif average_emotion_scores["neutral"] > 50:
            behavior = "The candidate is calm but emotionally distant."
            actions = "Ask more open-ended questions to engage the candidate emotionally."

    return behavior, actions

def generate_report(mode):
    global emotion_totals, frame_count
    report_filename = f"emotion_analysis_report_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    if frame_count == 0:
        frame_count = 1

    average_emotion_scores = {emotion: score / frame_count for emotion, score in emotion_totals.items()}

    with open(report_filename, 'w') as report_file:
        report_file.write("Emotion & Behavior Analysis Report\n")
        report_file.write("=" * 40 + "\n\n")

        report_file.write(f"Analysis Mode: {mode}\n\n")
        for emotion, score in average_emotion_scores.items():
            report_file.write(f"{emotion.capitalize()}: {score:.2f}%\n")

        behavior, actions = analyze_behavior(average_emotion_scores, mode)

        report_file.write("\nBehavioral Analysis: \n")
        report_file.write(f"Behavior: {behavior}\n")
        report_file.write(f"Suggested Action: {actions}\n")

        report_file.write("\nAnalysis Completed.")

    messagebox.showinfo("Report Generated", f"Analysis report saved as {report_filename}")

def show_mode_selector():
    root = tk.Tk()
    root.title("Emotion & Behavior Analyzer")

    label = tk.Label(root, text="Select the Action for Behavior Analysis", font=("Arial", 14))
    label.pack(pady=20)

    def on_mode_selected(mode):
        root.destroy()
        start_analysis(mode)

    detective_button = tk.Button(root, text="Detective", font=("Arial", 12), command=lambda: on_mode_selected("Detective"))
    detective_button.pack(pady=10)

    student_button = tk.Button(root, text="Student Behavior", font=("Arial", 12), command=lambda: on_mode_selected("Student Behavior"))
    student_button.pack(pady=10)

    interview_button = tk.Button(root, text="Interview", font=("Arial", 12), command=lambda: on_mode_selected("Interview"))
    interview_button.pack(pady=10)

    root.mainloop()

# Start the program by showing the mode selector
show_mode_selector()