# AI Assistant for Speech Recognition and OpenAI Chat Completion

This code provides an AI assistant that performs speech recognition and uses OpenAI's Chat Completion API for generating responses. It utilizes various libraries and models, including Hugging Sound, Transformers, and PyAudio.

## Installation

To run this code, you need to install the required packages and models. Follow the steps below:

1. Install the necessary Python packages:

```pip install openai huggingsound pynput pyaudio wave transformers datasets torch sounddevice soundfile```


2. Set up the OpenAI API key:
- Replace `YOUR_API_KEY` with your OpenAI API key in the code.

The models will be downloaded and cached once you run the code.

## Usage

1. Run the code by executing the following command:
This may take a while when you run it the first time, since the models will be downloaded.

python main.py

2. Press the spacebar to start/stop recording.

3. The assistant will transcribe the recorded audio using the speech recognition model and generate responses using the OpenAI Chat Completion API.

4. The assistant's responses will be converted to speech and played back.

## Customize

- You can modify the parameters and settings in the code to customize the behavior of the AI assistant, such as the maximum history length, temperature, and maximum tokens.

- Refer to the documentation of the used libraries and models for more details on customization options.

## Notes

- Make sure you have a stable internet connection to access the OpenAI API.

- This code assumes the audio input is mono and has a sample rate of 44100 Hz. Adjust the parameters if needed for your specific setup.

- For any issues or questions, please refer to the repository's [issue tracker](https://github.com/dev-doshi/Puja/issues).

---
