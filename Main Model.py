import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
import datetime
from deepface import DeepFace
import mediapipe as mp
from google import genai
import json

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


# Function to start analysis based on the selected mode and input source
def start_analysis(mode, input_source):
    global frame_count
    if input_source == "camera":
        cap = cv2.VideoCapture(0)
    else:
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if not file_path:
            return
        cap = cv2.VideoCapture(file_path)

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands, \
            mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            window_title = f"Emotion Analysis - {mode}"

            if not ret:
                break

            # Analyze the emotion using DeepFace
            emotion_scores = analyze_emotion(frame)

            # Accumulate the emotion percentages over time (for each frame)
            global emotion_totals
            for emotion, score in emotion_scores.items():
                emotion_totals[emotion] += score
            frame_count += 1

            # Analyze hand gestures
            analyze_hands(frame, hands)

            # Analyze eye direction and head movement
            analyze_face(frame, face_mesh)

            # Find the dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            dominant_emotion_score = emotion_scores[dominant_emotion]

            # Display the dominant emotion and its percentage
            emotion_text = f"{dominant_emotion}: {dominant_emotion_score:.2f}%"
            frame = cv2.putText(frame, emotion_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the percentages for all emotions
            y_offset = 100
            for emotion, score in emotion_scores.items():
                frame = cv2.putText(frame, f"{emotion}: {score:.2f}%", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 0), 2)
                y_offset += 30

            # Mode-based behavior analysis
            if mode == "Detective":
                log_to_report(mode, f"Detecting: {dominant_emotion} ({dominant_emotion_score:.2f}%)")
            elif mode == "Student Behavior":
                log_to_report(mode, f"Tracking: {dominant_emotion} ({dominant_emotion_score:.2f}%)")
            elif mode == "Interview":
                log_to_report(mode, f"Analyzing: {dominant_emotion} ({dominant_emotion_score:.2f}%)")

            # Display the frame with the predicted emotion and percentages
            cv2.imshow(window_title, frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Generate the report based on accumulated emotions and body language
    generate_report(mode)


# Function to generate the report and display it in a GUI
def generate_report(mode):
    global emotion_totals, frame_count, hand_gesture_count, eye_direction_count, head_movement_count

    if frame_count == 0:  # Avoid division by zero
        frame_count = 1

    # Calculate the average percentage for each emotion
    average_emotion_scores = {emotion: score / frame_count for emotion, score in emotion_totals.items()}

    # Create a new window for the report
    report_window = tk.Tk()
    report_window.title("Emotion & Behavior Analysis Report")
    report_window.geometry("1000x600")

    # Add a ScrolledText widget to display the report
    report_text = scrolledtext.ScrolledText(report_window, wrap=tk.WORD, width=70, height=30)
    report_text.pack(padx=10, pady=10)

    # Write the report content
    report_text.insert(tk.END, "Emotion & Behavior Analysis Report\n")
    report_text.insert(tk.END, "=" * 40 + "\n\n")

    # Write the emotion percentages
    report_text.insert(tk.END, f"Analysis Mode: {mode}\n\n")
    for emotion, score in average_emotion_scores.items():
        report_text.insert(tk.END, f"{emotion.capitalize()}: {score:.2f}%\n")

    # Write the body language analysis
    report_text.insert(tk.END, "\nBody Language Analysis:\n")
    report_text.insert(tk.END, f"Tense Hand Gestures: {hand_gesture_count['tense']}\n")
    report_text.insert(tk.END, f"Relaxed Hand Gestures: {hand_gesture_count['relaxed']}\n")
    report_text.insert(tk.END, f"Eye Direction (Left): {eye_direction_count['left']}\n")
    report_text.insert(tk.END, f"Eye Direction (Right): {eye_direction_count['right']}\n")
    report_text.insert(tk.END, f"Eye Direction (Center): {eye_direction_count['center']}\n")
    report_text.insert(tk.END, f"Head Movement (Up): {head_movement_count['up']}\n")
    report_text.insert(tk.END, f"Head Movement (Down): {head_movement_count['down']}\n")
    report_text.insert(tk.END, f"Head Movement (Still): {head_movement_count['still']}\n")

    analysis = ""
    for emotion, score in average_emotion_scores.items():
        analysis += f"{emotion.capitalize()}: {score:.2f}%\n"

    analysis += f"Tense Hand Gestures: {hand_gesture_count['tense']}\nRelaxed Hand Gestures: {hand_gesture_count['relaxed']}\nEye Direction (Left): {eye_direction_count['left']}\nEye Direction (Right): {eye_direction_count['right']}\nEye Direction (Center): {eye_direction_count['center']}\nHead Movement (Up): {head_movement_count['up']}\nHead Movement (Down): {head_movement_count['down']}\nHead Movement (Still): {head_movement_count['still']}\n"

    # Behavioral analysis based on emotion percentages and body language
    behavior, actions = analyze_behavior(average_emotion_scores, mode)

    # Write the behavior and action suggestions
    report_text.insert(tk.END, "\nBehavioral Analysis: \n")

    # Gemini API call
    client = genai.Client(api_key="AIzaSyCEy8plT-PIyPtSpYOnxMVtLWgQnvRMUgo")
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=f"""Based on this Emotion & Behavior Analysis for a {mode}: {analysis}, 
    use this JSON format to structure the output:

    {{
        "behavior": "describe overall behavior based on the analysis",
        "action": "suggest appropriate action"
    }}
    """
    )

    ans = response.text
    ans = json.loads(ans[8:-4])
    report_text.insert(tk.END, f"Behavior: {ans['behavior']}\n\n")
    report_text.insert(tk.END, f"Suggested action: {ans['action']}\n\n")

    report_text.insert(tk.END, "\nAnalysis Completed.")

    # Add a "Return to Main Menu" button
    return_button = tk.Button(report_window, text="Return to Main Menu", font=("Arial", 12),
                              command=lambda: [report_window.destroy(), show_mode_selector()])
    return_button.pack(pady=10)

    # Show confirmation message
    messagebox.showinfo("Report Generated", "Analysis report is displayed in the new window.")


# Function to analyze behavior based on emotion percentages and body language
def analyze_behavior(average_emotion_scores, mode):
    global hand_gesture_count, eye_direction_count, head_movement_count
    behavior = ""
    actions = ""
    return behavior, actions


# Function to create the mode selection GUI
def show_mode_selector():
    # Create the main window
    root = tk.Tk()
    root.title("Emotion & Behavior Analyzer")
    root.geometry("1000x600")

    # Label for the instructions
    label = tk.Label(root, text="Select the Action for Behavior Analysis", font=("Arial", 14))
    label.pack(pady=20)

    # Function to handle the button click
    def on_mode_selected(mode):
        root.destroy()  # Close the GUI
        show_input_source_selector(mode)  # Show input source selection

    # Buttons for each mode
    detective_button = tk.Button(root, text="Detective", font=("Arial", 12),
                                 command=lambda: on_mode_selected("Criminal"))
    detective_button.pack(pady=10)

    student_button = tk.Button(root, text="Student Behavior", font=("Arial", 12),
                               command=lambda: on_mode_selected("Student Behavior"))
    student_button.pack(pady=10)

    interview_button = tk.Button(root, text="Interview", font=("Arial", 12),
                                 command=lambda: on_mode_selected("Interview"))
    interview_button.pack(pady=10)

    # Add a "Close Project" button
    close_button = tk.Button(root, text="Close Project", font=("Arial", 12), command=root.destroy)
    close_button.pack(pady=10)

    # Run the GUI main loop
    root.mainloop()


# Function to create the input source selection GUI
def show_input_source_selector(mode):
    # Create the main window
    root = tk.Tk()
    root.title("Select Input Source")
    root.geometry("1000x600")

    # Label for the instructions
    label = tk.Label(root, text="Select the Input Source for Analysis", font=("Arial", 14))
    label.pack(pady=20)

    # Function to handle the button click
    def on_input_source_selected(input_source):
        root.destroy()  # Close the GUI
        start_analysis(mode, input_source)  # Start the analysis based on selected mode and input source

    # Buttons for each input source
    camera_button = tk.Button(root, text="Use Camera", font=("Arial", 12),
                              command=lambda: on_input_source_selected("camera"))
    camera_button.pack(pady=10)

    video_button = tk.Button(root, text="Upload Video", font=("Arial", 12),
                             command=lambda: on_input_source_selected("video"))
    video_button.pack(pady=10)

    # Run the GUI main loop
    root.mainloop()


# Start the program by showing the mode selector
show_mode_selector()
