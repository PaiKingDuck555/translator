import os
import sys
import asyncio
import tempfile
import subprocess
from pathlib import Path

from transformers import MarianMTModel, MarianTokenizer
import numpy as np

# ── Optional audio imports (graceful if missing) ──
AUDIO_AVAILABLE = False
TTS_AVAILABLE = False

try:
    import sounddevice as sd
    from scipy.io.wavfile import write as write_wav
    AUDIO_AVAILABLE = True
except (OSError, ImportError):
    print("⚠️  Audio recording unavailable (PortAudio/sounddevice not found).")
    print("   Text-only mode will still work.\n")

try:
    import edge_tts
    TTS_AVAILABLE = True
except ImportError:
    print("⚠️  Text-to-speech unavailable (edge-tts not found).\n")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  Whisper unavailable (openai-whisper not found).\n")


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
RECORD_SECONDS = 7   # Max recording duration per clip

# Edge-TTS voice options (natural-sounding)
VOICES = {
    "en": "en-US-ChristopherNeural",   # English voice
    "es": "es-MX-JorgeNeural",         # Spanish voice
}

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
}


# ──────────────────────────────────────────────
# 1. SPEECH-TO-TEXT  (OpenAI Whisper)
# ──────────────────────────────────────────────
def load_whisper_model(size="base"):
    """Load Whisper model. Sizes: tiny, base, small, medium, large"""
    if not WHISPER_AVAILABLE:
        print("⚠️  Whisper not installed — skipping.")
        return None
    print(f"Loading Whisper ({size}) model...")
    return whisper.load_model(size)


def record_audio(duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
    """Record audio from microphone."""
    if not AUDIO_AVAILABLE:
        print("❌ Cannot record: PortAudio/sounddevice not available.")
        print("   Install with: sudo apt install libportaudio2")
        return None

    print(f"🎤 Recording for {duration} seconds... (speak now!)")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                   channels=1, dtype="float32")
    sd.wait()  # Wait until recording is finished
    print("✅ Recording complete.")
    return audio.flatten()


def transcribe_audio(whisper_model, audio_data=None, audio_file=None):
    """Transcribe audio to text using Whisper."""
    if whisper_model is None:
        print("❌ Whisper model not loaded.")
        return "", "en"

    if audio_file:
        result = whisper_model.transcribe(audio_file)
    elif audio_data is not None:
        # Save to temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            write_wav(f.name, SAMPLE_RATE, (audio_data * 32767).astype(np.int16))
            temp_path = f.name
        result = whisper_model.transcribe(temp_path)
        os.unlink(temp_path)
    else:
        raise ValueError("Provide either audio_data or audio_file")

    return result["text"].strip(), result.get("language", "en")


# ──────────────────────────────────────────────
# 2. TRANSLATION  (MarianMT)
# ──────────────────────────────────────────────
def load_translation_models():
    """Load translation models for English <-> Spanish."""
    models = {}

    print("Loading English → Spanish model...")
    models["en-es"] = {
        "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es"),
        "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es"),
    }

    print("Loading Spanish → English model...")
    models["es-en"] = {
        "model": MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en"),
        "tokenizer": MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en"),
    }

    return models


def translate_text(text, models, source_lang, target_lang):
    """Translate text between languages."""
    key = f"{source_lang}-{target_lang}"
    if key not in models:
        print(f"⚠️  No model for {key}, returning original text.")
        return text

    model = models[key]["model"]
    tokenizer = models[key]["tokenizer"]

    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


# ──────────────────────────────────────────────
# 3. TEXT-TO-SPEECH  (Edge TTS)
# ──────────────────────────────────────────────
async def text_to_speech(text, lang="es", output_file="output.mp3"):
    """Convert text to speech using edge-tts."""
    if not TTS_AVAILABLE:
        print("❌ edge-tts not available.")
        return None
    voice = VOICES.get(lang, VOICES["en"])
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    return output_file


def speak(text, lang="es"):
    """Synchronous wrapper to generate and play TTS audio."""
    if not TTS_AVAILABLE:
        print("❌ Text-to-speech not available.")
        return

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        output_file = f.name

    asyncio.run(text_to_speech(text, lang, output_file))

    # Play the audio file
    play_audio(output_file)
    os.unlink(output_file)


def play_audio(filepath):
    """Play an audio file using available system player."""
    system = sys.platform
    try:
        if system == "darwin":  # macOS
            subprocess.run(["afplay", filepath], check=True)
        elif system == "linux":
            # Try common Linux audio players
            for player in ["mpv", "ffplay", "aplay", "paplay"]:
                if subprocess.run(["which", player], capture_output=True).returncode == 0:
                    if player == "ffplay":
                        subprocess.run([player, "-nodisp", "-autoexit", filepath],
                                       check=True, capture_output=True)
                    else:
                        subprocess.run([player, filepath], check=True, capture_output=True)
                    return
            print(f"⚠️  No audio player found. Audio saved to: {filepath}")
        else:
            os.startfile(filepath)  # Windows
    except Exception as e:
        print(f"⚠️  Could not play audio: {e}")
        print(f"   Audio file saved at: {filepath}")


# ──────────────────────────────────────────────
# FULL PIPELINE
# ──────────────────────────────────────────────
def voice_translate(whisper_model, translation_models, source_lang, target_lang):
    """Full pipeline: Voice → Text → Translate → Voice"""
    if not AUDIO_AVAILABLE:
        print("\n❌ Voice mode requires a microphone + PortAudio.")
        print("   Install: sudo apt install libportaudio2")
        print("   Then connect a USB microphone.\n")
        return

    if whisper_model is None:
        print("\n❌ Whisper model not loaded. Cannot do voice mode.\n")
        return

    src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)

    # Step 1: Record & transcribe
    print(f"\n🎤 Speak in {src_name}...")
    audio = record_audio()
    if audio is None:
        return
    text, detected_lang = transcribe_audio(whisper_model, audio_data=audio)
    print(f"📝 You said: \"{text}\"")

    if not text:
        print("⚠️  No speech detected. Try again.")
        return

    # Step 2: Translate
    translated = translate_text(text, translation_models, source_lang, target_lang)
    print(f"🔄 {tgt_name}: \"{translated}\"")

    # Step 3: Speak translation
    print(f"🔊 Speaking in {tgt_name}...")
    speak(translated, lang=target_lang)
    print("✅ Done!\n")


def text_only_mode(translation_models):
    """Text-only translation mode (no audio)."""
    print("\n=== Text Mode ===")
    print("Prefix with 'en:' or 'es:' to set the source language.")
    print("  en: I love programming  →  translates to Spanish")
    print("  es: Me encanta programar  →  translates to English")
    print("Type 'quit' to return to menu.\n")

    while True:
        user_input = input("> ").strip()

        if user_input.lower() == "quit":
            break

        if user_input.lower().startswith("en:"):
            text = user_input[3:].strip()
            result = translate_text(text, translation_models, "en", "es")
            print(f"🇪🇸 Spanish: {result}\n")

        elif user_input.lower().startswith("es:"):
            text = user_input[3:].strip()
            result = translate_text(text, translation_models, "es", "en")
            print(f"🇺🇸 English: {result}\n")

        else:
            print("Please prefix with 'en:' or 'es:'.\n")


def main():
    print("=" * 50)
    print("  🌐 Voice Translator")
    print("  English ↔ Spanish")
    print("=" * 50)

    # Load models
    whisper_model = load_whisper_model("base")
    translation_models = load_translation_models()
    print("\n✅ Models loaded!\n")

    # Show capability summary
    print("Capabilities:")
    print(f"  Translation (text): ✅ Ready")
    print(f"  Speech-to-text:     {'✅ Ready' if whisper_model else '❌ Unavailable'}")
    print(f"  Microphone input:   {'✅ Ready' if AUDIO_AVAILABLE else '❌ No PortAudio/mic'}")
    print(f"  Text-to-speech:     {'✅ Ready' if TTS_AVAILABLE else '❌ No edge-tts'}")
    print()

    while True:
        print("Choose a mode:")
        if AUDIO_AVAILABLE and whisper_model:
            print("  [1] 🎤 Voice: English → Spanish")
            print("  [2] 🎤 Voice: Spanish → English")
        else:
            print("  [1] 🎤 Voice: English → Spanish  (unavailable)")
            print("  [2] 🎤 Voice: Spanish → English  (unavailable)")
        print("  [3] ⌨️  Text-only translation")
        print("  [4] 🚪 Quit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            voice_translate(whisper_model, translation_models, "en", "es")
        elif choice == "2":
            voice_translate(whisper_model, translation_models, "es", "en")
        elif choice == "3":
            text_only_mode(translation_models)
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice. Try again.\n")


if __name__ == "__main__":
    main()
