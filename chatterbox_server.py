from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import re
import json
import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

app = FastAPI(title="Chatterbox TTS Engine")

# 1. Load the core service configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "chatterbox_config.json")
try:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
except Exception:
    config = {
        "server": {"host": "0.0.0.0", "port": 8001},
        "settings": {"max_chars_per_chunk": 150, "pause_duration_seconds": 1.0}
    }

print("Loading Chatterbox Turbo model into RTX 3050 Ti VRAM...")
model = ChatterboxTurboTTS.from_pretrained(device="cuda")

# 2. Updated request model to handle line-by-line pipeline tracking
class TTSRequest(BaseModel):
    show_name: str          # e.g., "NewOldRadioShow"
    line_filename: str      # e.g., "line_001.wav" (passed directly from the parser array)
    text: str
    voice_filename: str     # e.g., "male_04.wav"
    exaggeration: float = 0.6  
    cfg_weight: float = 0.5 
    speed_factor: float = 1.0      # 1.0 = normal, 1.3 = faster, 0.75 = slower   

def clean_and_split_text(text, max_chars=150):
    """Splits an individual line into chunks at punctuation boundaries if it exceeds limits."""
    tokens = re.split(r'(?<=[.!?])\s+|\[pause\]', text.strip())
    chunks = []
    for token in tokens:
        token = token.strip()
        if token:
            chunks.append(token)
    return chunks

def apply_tempo(wav_path, speed_factor):
    if speed_factor == 1.0:
        return  # skip processing if no change needed
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    y_stretched = librosa.effects.time_stretch(y, rate=speed_factor)
    sf.write(wav_path, y_stretched, sr)

@app.post("/generate")
async def generate_audio(payload: TTSRequest):
    # 3. Dynamic Registry Lookup
    registry_path = os.path.join(os.path.dirname(__file__), "show_configs.json")
    if not os.path.exists(registry_path):
        raise HTTPException(status_code=500, detail="Central show_configs.json registry file is missing.")
        
    with open(registry_path, "r") as f:
        show_registry = json.load(f)
        
    if payload.show_name not in show_registry:
        raise HTTPException(status_code=404, detail=f"Show configuration '{payload.show_name}' not found in registry.")
        
    show_config = show_registry[payload.show_name]
    
    # 4. Route to the temporary workspace directory for line staging
    voice_folder = show_config["voice_folder"]
    workspace_folder = show_config["workspace_folder"]
    
    speaker_path = os.path.join(voice_folder, payload.voice_filename)
    if not os.path.exists(speaker_path):
        raise HTTPException(status_code=404, detail=f"Voice file not found at: {speaker_path}")
        
    # Ensure workspace directory exists
    os.makedirs(workspace_folder, exist_ok=True)
    final_output_path = os.path.join(workspace_folder, payload.line_filename)
    
    try:
        max_chars = config["settings"].get("max_chars_per_chunk", 150)
        chunks = clean_and_split_text(payload.text, max_chars=max_chars)
        combined_wavs = []
        
        pause_sec = config["settings"].get("pause_duration_seconds", 1.0)
        silence_length = int(model.sr * pause_sec)
        silence_tensor = torch.zeros((1, silence_length), device="cuda" if torch.cuda.is_available() else "cpu")

        for i, chunk in enumerate(chunks):
            wav = model.generate(
                text=chunk,
                audio_prompt_path=speaker_path,
                exaggeration=payload.exaggeration,
                cfg_weight=payload.cfg_weight
            )
            combined_wavs.append(wav)
            
            # Match the exact device of the generated audio dynamically (CPU or CUDA)
            if i < len(chunks) - 1:
                silence_length = int(model.sr * pause_sec)
                silence_tensor = torch.zeros((1, silence_length), device=wav.device)
                combined_wavs.append(silence_tensor)
            
        final_wav = torch.cat(combined_wavs, dim=-1)
        ta.save(final_output_path, final_wav, model.sr)
        apply_tempo(final_output_path, payload.speed_factor)
        
        return {
            "status": "success",
            "saved_to": final_output_path,
            "filename": payload.line_filename,
            "chunks_processed": len(chunks)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    host_ip = config["server"].get("host", "0.0.0.0")
    port_num = config["server"].get("port", 8001)
    uvicorn.run(app, host=host_ip, port=port_num)