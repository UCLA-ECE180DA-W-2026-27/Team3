import os
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import pyaudio
import whisper
import google.generativeai as genai


RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "6"))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
MIC_DEVICE_INDEX = int(os.getenv("MIC_DEVICE_INDEX", "0"))
CONTEXT_FILE = Path(__file__).resolve().with_name("context.txt")


def record_audio(output_path: Path, seconds: int = RECORD_SECONDS, rate: int = SAMPLE_RATE) -> None:
    """Record audio from the Raspberry Pi microphone and save it as a WAV file."""
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        input=True,
        input_device_index=MIC_DEVICE_INDEX,
        frames_per_buffer=1024,
    )

    frames = []
    print(f"Listening for {seconds} seconds...")
    for _ in range(0, int(rate / 1024 * seconds)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wav_file.setframerate(rate)
        wav_file.writeframes(b"".join(frames))


def load_whisper_model():
    """Load the Whisper model once so it can be reused across loop iterations."""
    model_name = os.getenv("WHISPER_MODEL", "base")
    return whisper.load_model(model_name)


def transcribe_audio(audio_path: Path, whisper_model) -> str:
    """Transcribe the recorded audio with OpenAI Whisper."""
    result = whisper_model.transcribe(str(audio_path), fp16=False)
    return result["text"].strip()


def load_context() -> str:
    """Load knowledge-base text from a file in this folder."""
    if CONTEXT_FILE.exists():
        return CONTEXT_FILE.read_text(encoding="utf-8")
    return "No local context file was found. Answer based only on the user's question."


def load_llm_model():
    """Create the Google Gemini model once for repeated use."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set GEMINI_API_KEY in your environment.")

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    return genai.GenerativeModel(model_name)


def ask_llm(question: str, context: str, llm_model) -> str:
    """Send the question plus local context to Gemini and return the answer."""
    prompt = (
        "You are a helpful assistant. Use the knowledge base below to answer the user's question. "
        "If the answer is not in the provided knowledge base, say so clearly.\n\n"
        f"Knowledge base:\n{context}\n\nUser question:\n{question}"
    )

    response = llm_model.generate_content(prompt)
    return response.text.strip()


def speak_text(text: str) -> None:
    """Play the answer through a speaker connected to the Raspberry Pi."""
    if shutil.which("espeak-ng"):
        subprocess.run(["espeak-ng", "-ven+f3", "-s150", text], check=False)
        return

    if shutil.which("espeak"):
        subprocess.run(["espeak", "-ven+f3", "-s150", text], check=False)
        return

    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return
    except Exception:
        print(f"Speech output: {text}")


def main() -> None:
    whisper_model = load_whisper_model()
    llm_model = load_llm_model()
    context = load_context()

    print("Voice assistant started. Say something, or say 'exit' to stop.")

    while True:
        temp_dir = Path(tempfile.gettempdir())
        audio_file = temp_dir / f"voice_input_{int(time.time())}.wav"

        try:
            record_audio(audio_file)
            transcript = transcribe_audio(audio_file, whisper_model)

            if not transcript:
                print("No speech detected. Listening again...")
                continue

            print(f"Transcript: {transcript}")

            if transcript.lower().strip() in {"exit", "quit", "stop"}:
                print("Stopping voice assistant.")
                break

            answer = ask_llm(transcript, context, llm_model)
            print(f"Assistant answer: {answer}")
            speak_text(answer)
        finally:
            if audio_file.exists():
                audio_file.unlink()

        time.sleep(0.5)


if __name__ == "__main__":
    main()
