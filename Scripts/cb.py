import time
import warnings
import json
import openai
import os
from gtts import gTTS
from playsound import playsound
import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import soundfile

openai.api_key = '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'

def chatgpt_api(input_text):
    messages = [
    {"role": "system", "content": "You are a helpful assistant."}]
    
    if input_text:
        messages.append(
            {"role": "user", "content": input_text},
        )
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
    
    reply = chat_completion.choices[0].message.content
    return reply
def record_audio():
    fs = 44100  # Sample rate
    seconds = 10  # Duration of recording
    if os.path.exists('file.wav'):
        os.remove('file.wav')
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('file.wav', fs, myrecording)  # Save as WAV file 

    if os.path.exists('new.wav'):
        os.remove('new.wav')
    data, samplerate = soundfile.read('file.wav')
    soundfile.write('new.wav', data, samplerate, subtype='PCM_16')

def query(file_path):
    recognizer = sr.Recognizer()
    jackhammer = sr.AudioFile(file_path)
    with jackhammer as source:
        audio = recognizer.record(source)

    input_text=recognizer.recognize_google(audio,language='en-IN')

    output_text=chatgpt_api(input_text)
    return output_text

def playaudio(text):
    if os.path.exists('out.mp3'):
        os.remove('out.mp3')
    audiobj=gTTS(text=text,slow=False)
    audiobj.save('out.mp3')
    playsound('out.mp3')

