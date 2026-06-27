from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import re
import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

app = FastAPI(title="Chatterbox TTS Engine")

# Singleton pattern to load the model into VRAM once when server starts
print("Loading Chatterbox Turbo model into RTX 3050 Ti VRAM...")
model = ChatterboxTurboTTS.from_pretrained(device="cuda")

class TTSRequest(BaseModel):
    text: str
    voice_folder: str
    voice_filename: str
    output_folder: str
    output_filename: str

def split_text(text, max_chars=150):
    """Splits incoming text into small sentence chunks to safely protect 4GB VRAM."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

@app.post("/generate")
async def generate_audio(payload: TTSRequest):
    # 1. Resolve source speaker reference path
    speaker_path = os.path.join(payload.voice_folder, payload.voice_filename)
    if not os.path.exists(speaker_path):
        raise HTTPException(status_code=404, detail=f"Voice file not found at: {speaker_path}")
    
    # 2. Ensure target output directory exists on host machine
    os.makedirs(payload.output_folder, exist_ok=True)
    final_output_path = os.path.join(payload.output_folder, payload.output_filename)
    
    try:
        chunks = split_text(payload.text)
        combined_wavs = []
        
        # 3. Process line-by-line safely under VRAM threshold
        for chunk in chunks:
            wav = model.generate(
                text=chunk,
                audio_prompt_path=speaker_path,
                exaggeration=0.6,
                cfg_weight=0.5
            )
            combined_wavs.append(wav)
            
        # 4. Concatenate and write straight to destination path
        final_wav = torch.cat(combined_wavs, dim=-1)
        ta.save(final_output_path, final_wav, model.sr)
        
        return {
            "status": "success",
            "saved_to": final_output_path,
            "chunks_processed": len(chunks)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Bind to localhost on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)