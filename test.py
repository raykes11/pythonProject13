import os
import numpy as np
from TTS.tts.datasets import load_tts_samples

# print(type(np.float32))
# # custom formatter implementation
# def myformatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
#     """Assumes each line as ```<filename>|<transcription>```
#     """
#     txt_file = os.path.join(root_path, manifest_file)
#     print(txt_file)
#     items = []
#     speaker_name = "my_speaker"
#     with open(txt_file, "r", encoding="utf-8") as ttf:
#         for line in ttf:
#             cols = line.split("|")
#             print(cols)
#             wav_file = os.path.join(root_path, "wavs", cols[0])
#             text = cols[1]
#             items.append({"text":text, "audio_file":wav_file})
#     return items
#
#
# from TTS.tts.utils.text.phonemizers import ESpeak
# phonemizer = ESpeak("tr")
# print(phonemizer.phonemize("Bu Türkçe, bir örnektir.", separator="|"))

#
import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(torch.cuda.is_available())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
with open('text.txt','r',encoding='utf-8') as text:
    text = text.read()
    print(text)
    tts.tts_to_file(text=text, speaker_wav="datase9.wav", language="ru", file_path="output_354657.wav")
# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
# wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
# Text to speech to a file
# tts.tts_to_file(text=text, speaker_wav="datase9.wav", language="ru", file_path="output_354657.wav")
