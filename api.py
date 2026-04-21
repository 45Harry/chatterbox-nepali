import asyncio
import io
import json
import logging
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pathlib import Path
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatterbox Nepali Streaming API")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "results/t3_mtl_nepali_final.safetensors"

model_wrapper = None

@app.on_event("startup")
async def startup_event():
    global model_wrapper
    logger.info(f"Loading Nepali TTS Model on {DEVICE}...")

    # Load from local base_model directory
    base_model_path = Path("/app/base_model")
    logger.info(f"Loading model from local base_model: {base_model_path}")
    model_wrapper = ChatterboxMultilingualTTS.from_local(base_model_path, DEVICE)

    # Load fine-tuned checkpoint if it exists
    if Path(CHECKPOINT).exists():
        logger.info(f"Loading fine-tuned safetensors from: {CHECKPOINT}")
        resume_state = load_file(CHECKPOINT, device=DEVICE)

        # Clean state dict keys (strip prefixes)
        cleaned_state = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in resume_state.items()}
        model_wrapper.t3.load_state_dict(cleaned_state, strict=False)
    else:
        logger.warning(f"{CHECKPOINT} not found! Using base multilingual weights.")

    model_wrapper.t3.to(DEVICE).eval()
    logger.info("Model loaded successfully.")

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio.
    Expects a JSON message with:
    {
      "text": "Your text here",
      "language_id": "ne",
      "ref_audio": "path/to/reference/audio.wav",
      "chunk_size": 25
    }
    Continually yields 16-bit PCM Audio Bytes for the synthesized chunk as soon as it is generated.
    """
    await websocket.accept()
    logger.info("WebSocket connection established.")
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            
            text = request.get("text")
            if not text:
                continue
                
            language_id = request.get("language_id", "ne")
            ref_audio = request.get("ref_audio", "example_nepali_ref.wav") # define a default reference if None
            
            exaggeration = float(request.get("exaggeration", 0.5))
            temperature = float(request.get("temperature", 0.8))
            chunk_size = int(request.get("chunk_size", 20)) # Small chunks for fast turnaround
            
            logger.info(f"Starting generation for text: {text[:20]}... with chunk_size: {chunk_size}")
            
            # Loop through the yielded chunks
            for wav_chunk in model_wrapper.generate_stream(
                text=text,
                language_id=language_id,
                audio_prompt_path=ref_audio,
                exaggeration=exaggeration,
                temperature=temperature,
                chunk_size=chunk_size
            ):
                # Wav chunk is normally fp32 [-1.0, 1.0]. Convert to 16-bit PCM for real-time streaming
                wav_chunk = np.clip(wav_chunk, -1.0, 1.0)
                pcm_16 = (wav_chunk * 32767).astype(np.int16)
                
                # Send the PCM byte chunk
                await websocket.send_bytes(pcm_16.tobytes())
                
                # Allow the event loop to breathe
                await asyncio.sleep(0.001)

            # Signal End of Stream for the current sentence
            await websocket.send_json({"status": "completed"})
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"Error during WebSocket stream: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/v1/audio/speech")
@app.post("/tts")
async def tts_endpoint(request: Request):
    """
    OpenAI-compatible TTS endpoint for Telvox.
    Accepts text and generates speech using the local Chatterbox model.

    Expected input format (OpenAI-compatible):
    {
        "model": "telvox",
        "input": "Your text here",
        "voice": "ne-Np-SagarNeural" or "af_kore",
        "response_format": "wav" (default) or "mp3",
        "speed": 1.0
    }
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    text = body.get("input", body.get("text", ""))
    if not text:
        return JSONResponse({"error": "Missing 'input' or 'text' field"}, status_code=400)

    # Voice parameter - telvox uses voice, OpenAI uses voice
    voice = body.get("voice", "af_kore")

    # Language detection from voice name (telvox style: ne-NP-*)
    language_id = "ne"  # default to Nepali
    if voice.startswith("ne-"):
        language_id = "ne"
    elif voice.startswith("en-"):
        language_id = "en"

    # Get parameters
    exaggeration = body.get("exaggeration", 0.5)
    temperature = body.get("temperature", 0.8)
    speed = body.get("speed", 1.0)

    logger.info(f"Generating speech for: {text[:50]}... voice: {voice}, lang: {language_id}")

    # Generate audio using the local model
    try:
        # Use a default reference audio if no custom one is provided
        ref_audio = body.get("ref_audio", "samples/ref.wav")

        if not Path(ref_audio).exists():
            # Try a few default paths
            default_refs = [
                "samples/ref.wav",
                "samples/ref2.wav",
                "example_nepali_ref.wav",
            ]
            for p in default_refs:
                if Path(p).exists():
                    ref_audio = p
                    break

        model_wrapper.prepare_conditionals(ref_audio, exaggeration=exaggeration)

        # Generate audio
        wav_tensor = model_wrapper.generate(
            text=text,
            language_id=language_id,
            audio_prompt_path=None,  # Already prepared
            exaggeration=exaggeration,
            temperature=temperature,
        )

        # Convert to numpy
        wav = wav_tensor.squeeze().cpu().numpy()

        # Convert to 16-bit PCM WAV format
        wav_int16 = np.clip(wav, -1.0, 1.0) * 32767
        wav_int16 = wav_int16.astype(np.int16)

        # Create WAV file in memory
        from scipy.io import wavfile
        buffer = io.BytesIO()
        # Use sample rate from model wrapper
        wavfile.write(buffer, model_wrapper.sr, wav_int16)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"X-Voice": voice, "X-Language": language_id}
        )

    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
