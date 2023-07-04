# Import the necessary packages
import openai
from huggingsound import SpeechRecognitionModel
from pynput import keyboard
import pyaudio
import wave
import os
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import sounddevice as sd
import soundfile as sf

# Set up the speech recognition model parameters
SRMODEL = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")

# Set up openai request parameters
OPENAI_API_KEY = "YOUR_API_KEY"
HISTORY = [{"role": "system", "content": "You are a helpful assistant able to answer questions without a name. instead of AI you type A I. You shall type every number which is not a letter always as words, for example instead of 14 you say four-teen"}]
MAX_HISTORY = 5
OAMODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.9
MAX_TOKENS = 50

openai.api_key = OPENAI_API_KEY

# Set up audio recording parameters
CHUNK = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
OUTPUT_FILENAME = "temp/audio.mp3"

frames = []
recording = False

# Set up the text to speech model parameters
PROCESSOR = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
MODEL = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
VOCODER = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

def text_to_speech(input):

    # Prepare the input string

    input = PROCESSOR(text=input, return_tensors="pt")
    
    speech = MODEL.generate_speech(input["input_ids"], speaker_embeddings, vocoder=VOCODER)

    sf.write("speech.wav", speech.numpy(), samplerate=16000)

    data, fs = sf.read("speech.wav", dtype='float32')  
    sd.play(data, fs)
    status = sd.wait()

def make_openai_request(input):

    print(HISTORY + [{"role": "user", "content": input["transcription"]}])

    completion = openai.ChatCompletion.create(
        model=OAMODEL,  
        messages=HISTORY + [{"role": "user", "content": input["transcription"]}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    HISTORY.append({"role": "user", "content": input["transcription"]})
    HISTORY.append({"role": "system", "content": completion.choices[0].message["content"]})

    # Play the response 
    text_to_speech(completion.choices[0].message["content"])
    
    print(completion.choices[0].message)

def on_press(key):
    global recording
    if key == keyboard.Key.space:
        recording = True

def on_release(key):
    global recording
    if key == keyboard.Key.space:
        recording = False


def listen():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    # Start listening for push-to-talk button events
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print("Program Started. Waiting for Keyboard Input")
    while True:
        
        if recording:
            print("Listening")
            while recording:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            print("Saving Audio")
            save_audio(listener, stream, audio)
            transcriptions = SRMODEL.transcribe(["temp/audio.mp3"])
            for transcription in transcriptions:
                make_openai_request(transcription)
                print("Waiting for Keyboard Input")

                # Delete the audio file
                os.remove(OUTPUT_FILENAME)
                frames.clear()

                if len(HISTORY) > MAX_HISTORY:
                    # Delete the oldest history entry which by the user
                    del HISTORY[1]
                
        else:
            pass

def save_audio(listener, stream, audio):

    # Save the recorded audio to an MP3 file
    wave_file = wave.open(OUTPUT_FILENAME, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()

    print("Recording saved to", OUTPUT_FILENAME)


if __name__ == "__main__":
    listen()