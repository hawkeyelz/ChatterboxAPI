import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

print("Initializing Chatterbox Turbo on RTX 3050 Ti...")
# Load the lower-overhead Turbo model
model = ChatterboxTurboTTS.from_pretrained(device="cuda")

# Text using official native paralinguistic tags
text = "Hi there! [laugh] It is so good to see you again. [chuckle] We finally got this setup running."

print("Generating audio...")
# Generate the audio waveform using default voice
wav = model.generate(text, exaggeration=0.5, cfg_weight=0.5)

# Save the output file
ta.save("chatterbox_output.wav", wav, model.sr)
print("Success! Saved output to chatterbox_output.wav")