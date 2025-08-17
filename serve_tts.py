#!/usr/bin/env python3
"""
Chatterbox TTS Server for GLaDOS
Run with: python serve_chatterbox.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import io
from pydantic import BaseModel
import logging
from typing import Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GLaDOS TTS Server")

# Global model variable
model = None
GLADOS_VOICE_PATH = "jarvis_sample.wav"  # Update this path

class TTSRequest(BaseModel):
    text: str
    exaggeration: float = 0.1  # GLaDOS dramatic style
    cfg_weight: float = 0.3    # Deliberate pacing
    use_glados_voice: bool = True

@app.on_event("startup")
async def load_model():
    """Load the model on startup"""
    global model
    try:
        logger.info("Loading Chatterbox model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ChatterboxTTS.from_pretrained(device=device)
        logger.info(f"Model loaded successfully on: {device}")
        
        # Check if GLaDOS voice sample exists
        if os.path.exists(GLADOS_VOICE_PATH):
            logger.info(f"GLaDOS voice sample found at: {GLADOS_VOICE_PATH}")
        else:
            logger.warning(f"GLaDOS voice sample not found at: {GLADOS_VOICE_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Generate speech from text and return audio stream"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        logger.info(f"Generating speech for: {request.text[:50]}...")
        
        # Generate audio
        if request.use_glados_voice and os.path.exists(GLADOS_VOICE_PATH):
            wav = model.generate(
                request.text, 
                audio_prompt_path=GLADOS_VOICE_PATH,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight
            )
        else:
            logger.warning("Using default voice (GLaDOS voice not found or disabled)")
            wav = model.generate(
                request.text,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight
            )
        
        # Convert to audio stream in memory
        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)
        
        logger.info("Audio generation complete")
        
        # Return as streaming response
        return StreamingResponse(
            buffer, 
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=glados_speech.wav"}
        )
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if the service is running and model is loaded"""
    return {
        "status": "operational" if model is not None else "loading",
        "device": str(model.device) if model else "not loaded",
        "glados_voice_available": os.path.exists(GLADOS_VOICE_PATH)
    }

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "GLaDOS TTS Server",
        "endpoints": {
            "/tts": "POST - Generate speech from text",
            "/health": "GET - Service health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    # Run the server
    logger.info("Starting GLaDOS TTS Server...")
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Listen on all interfaces
        port=8001,
        log_level="info"
    )