import os
import sys
import asyncio
import tempfile
import subprocess
import time
import difflib

from transformers import MarianMTModel, MarianTokenizer
import numpy as np

# ── Optional audio imports (graceful if missing) ──
AUDIO_AVAILABLE = False
TTS_AVAILABLE = False
WHISPER_AVAILABLE = False
EMBEDDINGS_AVAILABLE = False

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
    print("⚠️  Whisper unavailable (openai-whisper not found).\n")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("⚠️  Sentence embeddings unavailable (sentence-transformers not found).")
    print("   Translation verification will be limited.\n")


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

# Maps detected language → target language for auto mode
AUTO_TRANSLATE_MAP = {
    "en": "es",  # If you speak English → translate to Spanish
    "es": "en",  # If you speak Spanish → translate to English
}

# Embedding similarity threshold
EMBEDDING_THRESHOLD = 0.80

# Critical words that change meaning when swapped
CRITICAL_WORDS = {
    "prepositions": {
        "in", "at", "on", "to", "from", "with", "without", "into", "onto",
        "by", "for", "of", "off", "out", "through", "under", "over", "between",
        "inside", "outside", "behind", "before", "after", "against",
    },
    "negations": {
        "not", "no", "never", "neither", "nor", "none", "nobody", "nothing",
        "nowhere", "don't", "doesn't", "didn't", "won't", "wouldn't", "can't",
        "cannot", "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
        "haven't", "hasn't", "hadn't",
    },
    "quantities": {
        "all", "every", "some", "any", "none", "few", "many", "most",
        "each", "both", "several", "enough", "only", "one", "two", "three",
        "four", "five", "six", "seven", "eight", "nine", "ten",
    },
    "pronouns": {
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "its", "our", "their",
        "mine", "yours", "hers", "ours", "theirs", "myself", "yourself",
    },
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


def load_embedding_model():
    """Load sentence embedding model for translation verification."""
    if not EMBEDDINGS_AVAILABLE:
        print("⚠️  sentence-transformers not installed — verification limited.")
        return None
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    return SentenceTransformer("all-MiniLM-L6-v2")


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
    """Transcribe audio to text using Whisper. Returns (text, language_code)."""
    if whisper_model is None:
        print("❌ Whisper model not loaded.")
        return "", "en"

    if audio_file:
        result = whisper_model.transcribe(audio_file)
    elif audio_data is not None:
        # Save to temp wav file for Whisper
        temp_path = tempfile.mktemp(suffix=".wav")
        write_wav(temp_path, SAMPLE_RATE, (audio_data * 32767).astype(np.int16))
        result = whisper_model.transcribe(temp_path)
        os.unlink(temp_path)
    else:
        raise ValueError("Provide either audio_data or audio_file")

    detected_lang = result.get("language", "en")
    text = result["text"].strip()
    return text, detected_lang


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
# 3. TRANSLATION VERIFICATION
# ──────────────────────────────────────────────
def verify_translation(original, translated, models, source_lang, target_lang,
                       embedding_model=None):
    """
    Verify translation quality using back-translation + two-layer checking.

    Layer 1: Embedding cosine similarity (semantic meaning)
    Layer 2: Critical word detection (prepositions, negations, etc.)

    Returns a dict with verification results.
    """
    t_start = time.time()

    # Step 1: Back-translate
    back_translated = translate_text(translated, models, target_lang, source_lang)

    # Normalize for comparison
    orig_lower = original.lower().strip().rstrip("?.!,;:")
    back_lower = back_translated.lower().strip().rstrip("?.!,;:")
    orig_words = orig_lower.split()
    back_words = back_lower.split()

    # ── Layer 1: Embedding similarity ──
    embedding_score = None
    if embedding_model is not None:
        embeddings = embedding_model.encode([orig_lower, back_lower])
        # Cosine similarity
        dot = np.dot(embeddings[0], embeddings[1])
        norm_a = np.linalg.norm(embeddings[0])
        norm_b = np.linalg.norm(embeddings[1])
        embedding_score = float(dot / (norm_a * norm_b))

    # ── Layer 2: Critical word detection ──
    critical_flags = []

    # Find words that changed
    orig_set = set(orig_words)
    back_set = set(back_words)
    added_words = back_set - orig_set     # Words that appeared
    removed_words = orig_set - back_set   # Words that disappeared
    changed_words = added_words | removed_words

    for category, word_set in CRITICAL_WORDS.items():
        changed_critical = changed_words & word_set
        if changed_critical:
            # Figure out what changed to what
            added_critical = added_words & word_set
            removed_critical = removed_words & word_set
            critical_flags.append({
                "category": category,
                "removed": removed_critical,  # Was in original, missing in back-translation
                "added": added_critical,       # Not in original, appeared in back-translation
            })

    # Also check for negation appearing/disappearing (most dangerous)
    negation_flipped = bool(
        (removed_words & CRITICAL_WORDS["negations"]) or
        (added_words & CRITICAL_WORDS["negations"])
    )

    # ── Generate word-level diff ──
    diff = list(difflib.ndiff(orig_words, back_words))
    word_changes = []
    for d in diff:
        if d.startswith("- "):
            word_changes.append(f'  - "{d[2:]}"')
        elif d.startswith("+ "):
            word_changes.append(f'  + "{d[2:]}"')

    # ── Determine overall verdict ──
    has_critical_flag = len(critical_flags) > 0
    embedding_ok = (embedding_score is None) or (embedding_score >= EMBEDDING_THRESHOLD)
    exact_match = (orig_lower == back_lower)

    if exact_match:
        verdict = "✅ High confidence — back-translation matches perfectly"
        confidence = "high"
    elif has_critical_flag or negation_flipped:
        verdict = "🚨 Warning — critical word change detected"
        confidence = "low"
    elif not embedding_ok:
        verdict = "⚠️ Caution — meaning may have shifted"
        confidence = "medium"
    else:
        verdict = "✅ Good — translation looks accurate"
        confidence = "high"

    t_elapsed = time.time() - t_start

    return {
        "original": original,
        "translated": translated,
        "back_translated": back_translated,
        "embedding_score": embedding_score,
        "critical_flags": critical_flags,
        "negation_flipped": negation_flipped,
        "word_changes": word_changes,
        "exact_match": exact_match,
        "verdict": verdict,
        "confidence": confidence,
        "time_seconds": t_elapsed,
    }


def print_verification(result, pipeline_start_time=None):
    """Pretty-print verification results."""
    print(f"\n  🔁 Back-check: \"{result['back_translated']}\"")

    if result["exact_match"]:
        print(f"  {result['verdict']}")
    else:
        # Show embedding score if available
        if result["embedding_score"] is not None:
            score_pct = result["embedding_score"] * 100
            print(f"  📊 Semantic similarity: {score_pct:.0f}%")

        # Show critical flags
        for flag in result["critical_flags"]:
            cat = flag["category"]
            removed = ", ".join(f'"{w}"' for w in flag["removed"]) if flag["removed"] else ""
            added = ", ".join(f'"{w}"' for w in flag["added"]) if flag["added"] else ""
            parts = []
            if removed:
                parts.append(f"lost: {removed}")
            if added:
                parts.append(f"gained: {added}")
            print(f"  🚨 [{cat}] {' / '.join(parts)}")

        if result["negation_flipped"]:
            print(f"  🚨 NEGATION CHANGED — meaning may be inverted!")

        # Show word diff
        if result["word_changes"]:
            print(f"  📝 Word changes:")
            for change in result["word_changes"]:
                print(f"    {change}")

        print(f"  {result['verdict']}")

    # Show timing
    print(f"  ⏱️  Verification: {result['time_seconds']:.2f}s", end="")
    if pipeline_start_time is not None:
        total_time = time.time() - pipeline_start_time
        print(f"  |  Total pipeline: {total_time:.2f}s")
    else:
        print()
    print()


# ──────────────────────────────────────────────
# 4. TEXT-TO-SPEECH  (Edge TTS)
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

    output_file = tempfile.mktemp(suffix=".mp3")
    try:
        asyncio.run(text_to_speech(text, lang, output_file))
        play_audio(output_file)
    finally:
        if os.path.exists(output_file):
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
def voice_translate_auto(whisper_model, translation_models, embedding_model):
    """Auto-detect language from speech, then translate to the other language."""
    if not AUDIO_AVAILABLE:
        print("\n❌ Voice mode requires a microphone + PortAudio.")
        print("   Install: sudo apt install libportaudio2")
        print("   Then connect a USB microphone.\n")
        return

    if whisper_model is None:
        print("\n❌ Whisper model not loaded. Cannot do voice mode.\n")
        return

    # Step 1: Record & transcribe (Whisper auto-detects language)
    print("\n🎤 Speak in English or Spanish...")
    audio = record_audio()
    if audio is None:
        return

    t_pipeline = time.time()
    text, detected_lang = transcribe_audio(whisper_model, audio_data=audio)

    if not text:
        print("⚠️  No speech detected. Try again.")
        return

    src_name = LANGUAGE_NAMES.get(detected_lang, detected_lang)
    print(f"📝 Detected {src_name}: \"{text}\"")

    # Step 2: Determine target language
    target_lang = AUTO_TRANSLATE_MAP.get(detected_lang)
    if target_lang is None:
        print(f"⚠️  Detected language '{detected_lang}' — only English and Spanish are supported.")
        return

    tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)

    # Step 3: Translate
    t_translate_start = time.time()
    translated = translate_text(text, translation_models, detected_lang, target_lang)
    t_translate = time.time() - t_translate_start
    print(f"🔄 {tgt_name}: \"{translated}\"  ({t_translate:.2f}s)")

    # Step 4: Verify translation
    result = verify_translation(text, translated, translation_models,
                                detected_lang, target_lang, embedding_model)
    print_verification(result, t_pipeline)

    # Step 5: Speak translation
    print(f"🔊 Speaking in {tgt_name}...")
    speak(translated, lang=target_lang)
    print("✅ Done!\n")


def voice_translate(whisper_model, translation_models, embedding_model,
                    source_lang, target_lang):
    """Manual pipeline: Voice → Text → Translate → Verify → Voice."""
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

    t_pipeline = time.time()
    text, detected_lang = transcribe_audio(whisper_model, audio_data=audio)
    print(f"📝 You said: \"{text}\"")

    if not text:
        print("⚠️  No speech detected. Try again.")
        return

    # Warn if Whisper detected a different language than expected
    if detected_lang != source_lang:
        det_name = LANGUAGE_NAMES.get(detected_lang, detected_lang)
        print(f"ℹ️  Note: Whisper detected {det_name}, but translating as {src_name} as selected.")

    # Step 2: Translate
    t_translate_start = time.time()
    translated = translate_text(text, translation_models, source_lang, target_lang)
    t_translate = time.time() - t_translate_start
    print(f"🔄 {tgt_name}: \"{translated}\"  ({t_translate:.2f}s)")

    # Step 3: Verify translation
    result = verify_translation(text, translated, translation_models,
                                source_lang, target_lang, embedding_model)
    print_verification(result, t_pipeline)

    # Step 4: Speak translation
    print(f"🔊 Speaking in {tgt_name}...")
    speak(translated, lang=target_lang)
    print("✅ Done!\n")


def text_only_mode(translation_models, embedding_model):
    """Text-only translation mode with verification."""
    print("\n=== Text Mode ===")
    print("Just type a sentence — language is auto-detected!")
    print("Or prefix with 'en:' or 'es:' to force the source language.")
    print("Type 'quit' to return to menu.\n")

    while True:
        user_input = input("> ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        # Check for explicit prefix
        if user_input.lower().startswith("en:"):
            text = user_input[3:].strip()
            source, target = "en", "es"
        elif user_input.lower().startswith("es:"):
            text = user_input[3:].strip()
            source, target = "es", "en"
        else:
            # Auto-detect language
            text = user_input
            source, target = detect_text_language(text)

        if not text:
            print("Please enter some text.\n")
            continue

        src_name = LANGUAGE_NAMES.get(source, source)
        tgt_name = LANGUAGE_NAMES.get(target, target)

        t_pipeline = time.time()
        translated = translate_text(text, translation_models, source, target)
        t_translate = time.time() - t_pipeline

        print(f"  [{src_name} → {tgt_name}]")
        print(f"  {tgt_name}: {translated}  ({t_translate:.2f}s)")

        # Verify
        result = verify_translation(text, translated, translation_models,
                                    source, target, embedding_model)
        print_verification(result, t_pipeline)


def detect_text_language(text):
    """Simple heuristic to detect if text is Spanish or English.
    Returns (source_lang, target_lang) tuple."""
    text_lower = text.lower()

    # Spanish indicators: accented characters + common Spanish words
    spanish_chars = set("áéíóúñ¿¡ü")
    spanish_words = {
        "el", "la", "los", "las", "un", "una", "es", "son", "está", "están",
        "yo", "tú", "él", "ella", "nosotros", "ellos", "que", "por", "para",
        "con", "como", "pero", "donde", "cuando", "hola", "buenos", "buenas",
        "gracias", "por favor", "si", "no", "muy", "también", "aquí",
        "tiene", "tengo", "quiero", "puedo", "necesito", "hacer", "este",
        "esta", "ese", "esa", "todo", "bien", "más", "menos", "día",
        "me", "te", "se", "le", "nos", "les", "mi", "su", "gusta",
        "gustaria", "encanta", "comer", "beber", "hablar", "casa",
    }

    # Check for Spanish-specific characters
    if any(c in spanish_chars for c in text_lower):
        return "es", "en"

    # Check for common Spanish words
    words = set(text_lower.split())
    spanish_matches = words & spanish_words
    if len(spanish_matches) >= 2:
        return "es", "en"

    # Default to English
    return "en", "es"


def main():
    print("=" * 50)
    print("  🌐 Voice Translator")
    print("  English ↔ Spanish")
    print("=" * 50)

    # Load models
    whisper_model = load_whisper_model("base")
    translation_models = load_translation_models()
    embedding_model = load_embedding_model()
    print("\n✅ Models loaded!\n")

    # Show capability summary
    print("Capabilities:")
    print(f"  Translation (text):  ✅ Ready")
    print(f"  Speech-to-text:      {'✅ Ready' if whisper_model else '❌ Unavailable'}")
    print(f"  Microphone input:    {'✅ Ready' if AUDIO_AVAILABLE else '❌ No PortAudio/mic'}")
    print(f"  Text-to-speech:      {'✅ Ready' if TTS_AVAILABLE else '❌ No edge-tts'}")
    print(f"  Verification:        {'✅ Full (embeddings)' if embedding_model else '⚠️ Basic (word-level only)'}")
    print()

    voice_ok = AUDIO_AVAILABLE and whisper_model is not None

    while True:
        print("Choose a mode:")
        tag = "" if voice_ok else "  (unavailable)"
        print(f"  [1] 🎤 Auto-detect: speak either language{tag}")
        print(f"  [2] 🎤 Voice: English → Spanish{tag}")
        print(f"  [3] 🎤 Voice: Spanish → English{tag}")
        print(f"  [4] ⌨️  Text-only translation (auto-detects language)")
        print(f"  [5] 🚪 Quit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            voice_translate_auto(whisper_model, translation_models, embedding_model)
        elif choice == "2":
            voice_translate(whisper_model, translation_models, embedding_model, "en", "es")
        elif choice == "3":
            voice_translate(whisper_model, translation_models, embedding_model, "es", "en")
        elif choice == "4":
            text_only_mode(translation_models, embedding_model)
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice. Try again.\n")


if __name__ == "__main__":
    main()
