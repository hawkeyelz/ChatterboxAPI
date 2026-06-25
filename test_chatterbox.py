import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

print("Initializing Chatterbox on RTX 3050 Ti...")
# Initialize the model directly using the official layout
model = ChatterboxTTS.from_pretrained(device="cuda")

# Define our text with the inline paralinguistic tag
text = "I cannot believe we finally got this running. [laugh] It feels amazing to start over from scratch!"

print("Generating audio...")
# Generate the audio waveform
# We'll stick to default settings first to watch the VRAM
wav = model.generate(text, exaggeration=0.5, cfg=0.5)

# Save the output using torchaudio
ta.save("chatterbox_output.wav", wav, model.sr)
print("Success! Saved output to chatterbox_output.wav")