import deepspeech
import numpy as np
import wave
import soundfile as sf
import torch
from transformers import pipeline, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import pyaudio
import librosa
import os

# Disable GPU warnings for TensorFlow and Huggingface
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Hide symlink warning

# Load DeepSpeech Model for Text-Based Transcription
MODEL_FILE = "deepspeech-0.9.3-models.pbmm"
SCORER_FILE = "deepspeech-0.9.3-models.scorer"
deepspeech_model = deepspeech.Model(MODEL_FILE)
deepspeech_model.enableExternalScorer(SCORER_FILE)

# Load Emotion Analysis Model for Text-Based Emotion Detection
emotion_analyzer_text = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Load Wav2Vec2 Model for Audio-Based Emotion Detection
MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)


# Function to Convert Audio to Text
def speech_to_text(audio_file):
    wf = wave.open(audio_file, 'rb')
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    text = deepspeech_model.stt(audio)
    return text


# Function to Analyze Emotion Based on Text
def analyze_emotion_text(text):
    result = emotion_analyzer_text(text)
    return result


# Function to Resample Audio to a Desired Sample Rate
def resample_audio(audio_file, target_sample_rate=16000):
    audio_input, sr = librosa.load(audio_file, sr=None)  # Load without resampling
    if sr != target_sample_rate:
        # Resample audio using librosa
        audio_input = librosa.resample(audio_input, orig_sr=sr, target_sr=target_sample_rate)
    return audio_input



# Function to Pad Audio if It's Too Short
def pad_audio(audio_input, target_length=16000):
    if len(audio_input) < target_length:
        padding = np.zeros(target_length - len(audio_input))
        audio_input = np.concatenate([audio_input, padding])
    return audio_input


# Function to Analyze Emotion Based on Audio (Wav2Vec2 for Emotion)
def analyze_audio_emotion_wav2vec2(audio_input):
    # Load and preprocess the audio file
    audio_input = pad_audio(audio_input)

    # Convert the audio input to float32 (to match the model's expected type)
    audio_input = audio_input.astype(np.float32)

    # Preprocess the audio
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)

    # Get predictions from the model
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get predicted emotion (Choose label according to the model's output)
    predicted_emotion = torch.argmax(logits, dim=-1)
    emotion_label = ['happy', 'sad', 'angry', 'neutral']  # Adjust this list according to the model's labels
    return emotion_label[predicted_emotion.item()]


# Function to Record Live Audio and Convert to Text
def live_audio_to_text(duration=10, samplerate=16000):
    print("Listening...")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=samplerate, input=True, frames_per_buffer=1024)
    frames = []
    for _ in range(0, int(samplerate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    text = deepspeech_model.stt(audio_data)
    return text, audio_data


# Main Execution
if __name__ == "__main__":
    mode = input("Choose mode: [1] Live Audio [2] File Input: ")

    if mode == '1':
        # Live Audio mode
        text, audio_data = live_audio_to_text(duration=10)  # 10 seconds of audio
        print(f"Transcribed Text: {text}")

        # Emotion Analysis Based on Text
        emotion_result_text = analyze_emotion_text(text)
        print(f"Emotion Analysis (Text-Based): {emotion_result_text}")

        # Emotion Analysis Based on Live Audio
        emotion_result_audio = analyze_audio_emotion_wav2vec2(audio_data)
        print(f"Emotion Analysis (Audio-Based): {emotion_result_audio}")

    elif mode == '2':
        # File input mode
        audio_path = input("Enter audio file path: ")
        text = speech_to_text(audio_path)
        print(f"Transcribed Text: {text}")

        # Emotion Analysis Based on Text
        emotion_result_text = analyze_emotion_text(text)
        print(f"Emotion Analysis (Text-Based): {emotion_result_text}")

        # Emotion Analysis Based on Audio
        audio_input = resample_audio(audio_path)
        emotion_result_audio = analyze_audio_emotion_wav2vec2(audio_input)
        print(f"Emotion Analysis (Audio-Based): {emotion_result_audio}")

    else:
        print("Invalid mode selected.")
        exit()
