from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import os
from init_chatbot import get_bot_response

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    return FileResponse("static/index.html")

@app.post("/chat/audio")
async def chat_via_voice(audio: UploadFile = File(...)):
    # Save the uploaded browser audio (likely webm/ogg)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(await audio.read())
        tmp_input_path = tmp.name

    tmp_wav_path = tmp_input_path.replace(".webm", ".wav")

    try:
        # Convert to WAV using pydub
        sound = AudioSegment.from_file(tmp_input_path)
        sound.export(tmp_wav_path, format="wav")

        # Transcribe
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_wav_path) as source:
            audio_data = recognizer.record(source)
            try:
                query = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                return JSONResponse(content={"error": "Could not understand audio."}, status_code=400)

        response = get_bot_response(query)
        return {"query": query, "response": response}

    finally:
        os.remove(tmp_input_path)
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)

@app.post("/chat/text")
async def chat_via_text(text: str = Form(...)):
    response = get_bot_response(text)
    return {"query": text, "response": response}
