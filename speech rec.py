import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load pre-trained model and processor
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Load audio (must be 16kHz, mono)
waveform, sample_rate = torchaudio.load("your_audio.wav")

# Resample if needed
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# Preprocess audio
inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)

# Perform inference
with torch.no_grad():
    logits = model(**inputs).logits

# Decode the predicted text
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]
print("Transcription:", transcription)
