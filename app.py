from fastapi import APIRouter, UploadFile, File
import whisper
import os
from gtts import gTTS
import base64
import tempfile
import requests
import streamlit as st  # Only if your backend is deployed with Streamlit

router = APIRouter()

# Load Whisper Model
model = whisper.load_model("base")

# Set up the API key using GOOGLE_API_KEY from environment or Streamlit secrets
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', st.secrets.get("GOOGLE_API_KEY"))
GEMINI_ENDPOINT = "https://api.gemini.com/v1/chat"  # Update with your actual Gemini endpoint

def get_gemini_response(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GOOGLE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7
    }
    response = requests.post(GEMINI_ENDPOINT, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    # Adjust based on the actual response format from Gemini
    return result.get("reply", "Sorry, no response was generated.")

@router.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        temp_audio_path = f"temp_{file.filename}"
        with open(temp_audio_path, "wb") as audio_file:
            audio_file.write(file.file.read())

        # Transcribe audio using Whisper
        transcription = model.transcribe(temp_audio_path)["text"]
        os.remove(temp_audio_path)  # Cleanup temporary file

        # Get conversational response from Gemini API using the transcription as the prompt
        ai_response_text = get_gemini_response(transcription)

        # Convert the Gemini text response to speech using gTTS
        tts = gTTS(text=ai_response_text, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
            ai_audio_path = temp_audio.name

        # Read the generated audio file and encode it to base64 for the frontend
        with open(ai_audio_path, "rb") as audio_file:
            ai_audio_bytes = audio_file.read()
        os.remove(ai_audio_path)  # Cleanup temporary file

        ai_audio_b64 = base64.b64encode(ai_audio_bytes).decode("utf-8")

        # Return both transcription and AI response (with audio)
        return {
            "transcription": transcription,
            "ai_response": ai_response_text,
            "ai_audio": ai_audio_b64
        }
    except Exception as e:
        return {"error": str(e)}
