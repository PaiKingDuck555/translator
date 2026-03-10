#!/usr/bin/env python3
"""
Edge Translator Device — Raspberry Pi 5
========================================

Standalone voice-to-voice translation device.
Press Enter (or push-to-talk button) → speak → press Enter (or release) → hear translation.

Hardware Setup:
  ┌─────────────────────────────────────────────────────┐
  │  USB MICROPHONE  → any USB port                     │
  │  USB SPEAKER     → any USB port (or 3.5mm jack)     │
  │                                                     │
  │  Optional GPIO (use --gpio flag):                   │
  │    BUTTON: GPIO 17 (pin 11) → GND (pin 9)          │
  │    GREEN LED: GPIO 27 (pin 13) → 330Ω → GND        │
  │    RED LED: GPIO 22 (pin 15) → 330Ω → GND          │
  └─────────────────────────────────────────────────────┘

Usage:
    python edge_device.py              # Keyboard mode (default)
    python edge_device.py --gpio       # GPIO push-to-talk mode
    python edge_device.py --list-audio # List available USB audio devices
    python edge_device.py --mic 1      # Use specific mic (device index)
    python edge_device.py --speaker 3  # Use specific speaker (device index)
"""

import os
import sys
import time
import argparse
import threading
import tty
import termios
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from scipy.signal import resample_poly
from math import gcd
import tempfile

# Import translation functions from our main module
from translate import (
    load_whisper_model,
    load_translation_models,
    load_embedding_model,
    load_tts_voices,
    transcribe_audio,
    translate_text,
    verify_translation,
    print_verification,
    speak,
    AUTO_TRANSLATE_MAP,
    LANGUAGE_NAMES,
)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
WHISPER_RATE = 16000     # Whisper expects 16kHz — we resample to this
CHANNELS = 1             # Mono audio
MAX_RECORD_SECONDS = 30  # Safety limit

# Directory to save recorded & generated audio for review
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")

# GPIO pins (BCM numbering) — only used with --gpio
BUTTON_PIN = 17
LED_GREEN = 27
LED_RED = 22


# ──────────────────────────────────────────────
# USB AUDIO HELPERS
# ──────────────────────────────────────────────
def list_audio_devices():
    """List all available audio devices with their index, name, and type."""
    devices = sd.query_devices()
    default_in, default_out = sd.default.device

    print("\n🎤 Available Audio Devices:")
    print("─" * 60)

    inputs = []
    outputs = []

    for i, d in enumerate(devices):
        is_input = d['max_input_channels'] > 0
        is_output = d['max_output_channels'] > 0
        tags = []
        if i == default_in:
            tags.append("DEFAULT INPUT")
        if i == default_out:
            tags.append("DEFAULT OUTPUT")
        tag_str = f"  ← {', '.join(tags)}" if tags else ""

        if is_input:
            inputs.append((i, d['name'], d['max_input_channels'], tag_str))
        if is_output:
            outputs.append((i, d['name'], d['max_output_channels'], tag_str))

    print("\n  📥 INPUT devices (microphones):")
    if inputs:
        for idx, name, ch, tag in inputs:
            print(f"     [{idx}] {name} ({ch}ch){tag}")
    else:
        print("     ⚠️  No input devices found! Plug in a USB mic.")

    print("\n  📤 OUTPUT devices (speakers):")
    if outputs:
        for idx, name, ch, tag in outputs:
            print(f"     [{idx}] {name} ({ch}ch){tag}")
    else:
        print("     ⚠️  No output devices found! Plug in a USB speaker.")

    print("─" * 60)
    return inputs, outputs


def get_mic_native_rate(mic_index):
    """
    Find the USB mic's native sample rate.

    USB mics don't all support the same sample rates. A cheap USB mic
    might only do 48kHz or 44.1kHz — if we try to open it at 16kHz
    (what Whisper wants), PortAudio throws PaErrorCode -9997.

    So we probe the device to find a rate it actually supports, then
    record at that rate and resample to 16kHz later.
    """
    info = sd.query_devices(mic_index)
    default_rate = int(info['default_samplerate'])

    # Try the device's advertised default rate first
    try:
        sd.check_input_settings(device=mic_index, samplerate=default_rate)
        return default_rate
    except Exception:
        pass

    # Try common sample rates (highest first for best quality before resample)
    for rate in [48000, 44100, 32000, 22050, 16000, 8000]:
        try:
            sd.check_input_settings(device=mic_index, samplerate=rate)
            return rate
        except Exception:
            continue

    # Last resort: return the advertised default and hope for the best
    return default_rate


def configure_audio_devices(mic_index=None, speaker_index=None):
    """
    Set the default sounddevice input/output devices.
    If indices are provided, use those; otherwise use system defaults.
    Returns (input_device_index, output_device_index, mic_sample_rate).
    """
    default_in, default_out = sd.default.device

    # Configure microphone (input)
    if mic_index is not None:
        try:
            info = sd.query_devices(mic_index)
            if info['max_input_channels'] == 0:
                print(f"⚠️  Device [{mic_index}] '{info['name']}' has no input channels.")
                print(f"   Falling back to default input [{default_in}].")
                mic_index = default_in
            else:
                print(f"🎤 Mic: [{mic_index}] {info['name']}")
        except Exception as e:
            print(f"⚠️  Invalid mic device index {mic_index}: {e}")
            print(f"   Falling back to default input [{default_in}].")
            mic_index = default_in
    else:
        mic_index = default_in
        if mic_index is not None:
            info = sd.query_devices(mic_index)
            print(f"🎤 Mic: [{mic_index}] {info['name']} (default)")
        else:
            print("⚠️  No default input device found! Plug in a USB mic.")

    # Detect mic's native sample rate
    mic_rate = WHISPER_RATE  # fallback
    if mic_index is not None:
        mic_rate = get_mic_native_rate(mic_index)
        if mic_rate != WHISPER_RATE:
            print(f"   Native rate: {mic_rate} Hz → will resample to {WHISPER_RATE} Hz for Whisper")
        else:
            print(f"   Native rate: {mic_rate} Hz (matches Whisper, no resampling needed)")

    # Configure speaker (output)
    if speaker_index is not None:
        try:
            info = sd.query_devices(speaker_index)
            if info['max_output_channels'] == 0:
                print(f"⚠️  Device [{speaker_index}] '{info['name']}' has no output channels.")
                print(f"   Falling back to default output [{default_out}].")
                speaker_index = default_out
            else:
                print(f"🔊 Speaker: [{speaker_index}] {info['name']}")
        except Exception as e:
            print(f"⚠️  Invalid speaker device index {speaker_index}: {e}")
            print(f"   Falling back to default output [{default_out}].")
            speaker_index = default_out
    else:
        speaker_index = default_out
        if speaker_index is not None:
            info = sd.query_devices(speaker_index)
            print(f"🔊 Speaker: [{speaker_index}] {info['name']} (default)")
        else:
            print("⚠️  No default output device found! Plug in a USB speaker.")

    # Apply to sounddevice defaults
    sd.default.device = (mic_index, speaker_index)

    return mic_index, speaker_index, mic_rate


class TranslatorDevice:
    """Standalone translation device with USB audio and optional GPIO."""

    def __init__(self, use_gpio=False, mic_index=None, speaker_index=None):
        self.use_gpio = use_gpio

        # Audio state
        self.audio_buffer = []
        self.is_recording = False
        self.is_processing = False
        self.mic_index = mic_index
        self.speaker_index = speaker_index
        self.mic_rate = WHISPER_RATE  # Will be updated after device detection
        self.block_size = 1600       # Recalculated based on mic_rate

        # Models (loaded at boot)
        self.whisper_model = None
        self.translation_models = None
        self.embedding_model = None

        # GPIO hardware (optional)
        self.button = None
        self.led_ready = None
        self.led_busy = None

        if self.use_gpio:
            self._init_gpio()

    # ─────────────────────────────────────
    # GPIO SETUP (optional, --gpio flag)
    # ─────────────────────────────────────
    def _init_gpio(self):
        """Initialize GPIO pins for button and LEDs."""
        try:
            from gpiozero import Button, LED
            self.button = Button(BUTTON_PIN, pull_up=True, bounce_time=0.05)
            self.led_ready = LED(LED_GREEN)
            self.led_busy = LED(LED_RED)
            print(f"✅ GPIO initialized:")
            print(f"   Button:    GPIO {BUTTON_PIN} (pin 11)")
            print(f"   Green LED: GPIO {LED_GREEN} (pin 13)")
            print(f"   Red LED:   GPIO {LED_RED} (pin 15)")
        except Exception as e:
            print(f"⚠️  GPIO init failed: {e}")
            print("   Falling back to keyboard mode.\n")
            self.use_gpio = False

    def set_state_ready(self):
        """Visual indicator: ready for input."""
        if self.led_ready:
            self.led_ready.on()
        if self.led_busy:
            self.led_busy.off()

    def set_state_recording(self):
        """Visual indicator: recording audio."""
        if self.led_ready:
            self.led_ready.off()
        if self.led_busy:
            self.led_busy.blink(on_time=0.1, off_time=0.1)

    def set_state_processing(self):
        """Visual indicator: processing translation."""
        if self.led_ready:
            self.led_ready.off()
        if self.led_busy:
            self.led_busy.on()

    def set_state_error(self):
        """Visual indicator: error occurred."""
        if self.led_ready:
            self.led_ready.off()
        if self.led_busy:
            self.led_busy.blink(on_time=0.5, off_time=0.5)

    # ─────────────────────────────────────
    # MODEL LOADING (runs once at boot)
    # ─────────────────────────────────────
    def load_models(self):
        """Load all AI models into memory. ~60-90s on Pi 5."""
        self.set_state_processing()
        print("\n" + "=" * 50)
        print("  🔄 Loading AI models (all local, no internet)...")
        print("=" * 50)

        t_start = time.time()

        self.whisper_model = load_whisper_model("base")
        self.translation_models = load_translation_models()
        self.embedding_model = load_embedding_model()
        self.tts_voices = load_tts_voices()

        t_elapsed = time.time() - t_start
        print(f"\n✅ All models loaded in {t_elapsed:.1f}s")
        print(f"   RAM usage: ~1.4 GB / 8 GB  (all offline)\n")

        self.set_state_ready()

    # ─────────────────────────────────────
    # AUDIO CAPTURE (runs in background)
    # ─────────────────────────────────────
    def audio_callback(self, indata, frames, time_info, status):
        """
        Called by sounddevice every ~100ms while the stream is open.
        Runs in a SEPARATE THREAD — must be fast, no heavy processing.

        indata: numpy array of shape (block_size, CHANNELS)
                Each value is a float32 between -1.0 and 1.0
                representing the audio waveform amplitude.

        We just copy it into our buffer list. That's it.
        """
        # Silently ignore overflow — it just means we lost a chunk while idle
        # (this is expected since the stream stays open between recordings)

        if self.is_recording:
            self.audio_buffer.append(indata.copy())

            # Safety limit: stop if recording too long
            chunks_recorded = len(self.audio_buffer)
            seconds_recorded = chunks_recorded * self.block_size / self.mic_rate
            if seconds_recorded >= MAX_RECORD_SECONDS:
                print(f"\n⚠️  Max recording time ({MAX_RECORD_SECONDS}s) reached.")
                self.is_recording = False

    # ─────────────────────────────────────
    # RECORDING CONTROL
    # ─────────────────────────────────────
    def start_recording(self):
        """Called when button is PRESSED (or Enter in keyboard mode)."""
        if self.is_processing:
            return  # Don't record while processing previous input

        self.audio_buffer = []
        self.is_recording = True
        self.set_state_recording()
        if self.use_gpio:
            print("🎤 Recording... (release button to stop)")
        else:
            print("🎤 Recording... (press [r] to stop)")

    def stop_recording(self):
        """Called when button is RELEASED (or Enter again in keyboard mode)."""
        if not self.is_recording:
            return

        self.is_recording = False
        chunk_count = len(self.audio_buffer)
        duration = chunk_count * self.block_size / self.mic_rate
        print(f"⏹️  Stopped. Captured {duration:.1f}s of audio ({chunk_count} chunks).")

        if chunk_count == 0:
            print("⚠️  No audio captured.\n")
            self.set_state_ready()
            return

        # Process in a separate thread so GPIO/keyboard stays responsive
        self.is_processing = True
        self.set_state_processing()

        # Concatenate all audio chunks into one array
        full_audio = np.concatenate(self.audio_buffer, axis=0).flatten()
        # Clear buffer to free memory
        self.audio_buffer = []

        # ── Resample to 16kHz for Whisper if mic uses a different native rate ──
        #
        # Most USB mics record at 48kHz (48,000 samples/sec) but Whisper
        # expects 16kHz. We can't just drop every 3rd sample because
        # frequencies above 8kHz (the Nyquist limit for 16kHz) would
        # "fold back" into the audible range as aliasing artifacts.
        #
        # resample_poly does three things:
        #   1. Upsample by 'up' factor (insert zeros between samples)
        #   2. Low-pass filter (removes frequencies above 8kHz so they
        #      can't alias — this is the critical step)
        #   3. Downsample by 'down' factor (keep every Nth sample)
        #
        # Example for 48kHz → 16kHz:
        #   GCD(48000, 16000) = 8000
        #   up   = 16000 / 8000 = 2   (upsample ×2 → 96kHz)
        #   down = 48000 / 8000 = 6   (downsample ÷6 → 16kHz)
        #   Net ratio: 2/6 = 1/3 — keeps 1 of every 3 samples, safely.
        #
        # Human speech lives below ~4kHz so cutting at 8kHz loses nothing.
        #
        if self.mic_rate != WHISPER_RATE:
            print(f"   Resampling {self.mic_rate} Hz → {WHISPER_RATE} Hz...")
            divisor = gcd(self.mic_rate, WHISPER_RATE)
            up = WHISPER_RATE // divisor
            down = self.mic_rate // divisor
            full_audio = resample_poly(full_audio, up, down).astype(np.float32)

        # Run processing in background thread
        thread = threading.Thread(target=self._process_and_reset,
                                  args=(full_audio,))
        thread.start()

    def _process_and_reset(self, audio_data):
        """Process audio and reset state when done."""
        try:
            self.process_audio(audio_data)
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            self.set_state_error()
            time.sleep(2)
        finally:
            self.is_processing = False
            self.set_state_ready()
            if self.use_gpio:
                print("🟢 Ready! Hold button to speak.\n")
            else:
                print("🟢 Ready! Press [r] to speak.\n")

    # ─────────────────────────────────────
    # PROCESSING PIPELINE
    # ─────────────────────────────────────
    def process_audio(self, audio_data):
        """
        Full pipeline: audio → text → translate → verify → speak

        This is where all the AI models run sequentially:
        1. Whisper:   audio_data (numpy array) → text + detected language
        2. MarianMT:  text → translated text
        3. MiniLM:    verify translation quality
        4. Piper TTS: translated text → audio file → play through USB speaker

        Audio files are saved to recordings/ for review.
        """
        t_pipeline = time.time()
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # ── Save input audio for review ──
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        input_wav = os.path.join(RECORDINGS_DIR, f"{timestamp}_input.wav")
        write_wav(input_wav, WHISPER_RATE, (audio_data * 32767).astype(np.int16))
        print(f"\n💾 Input saved: {input_wav}")

        # ── Step 1: Transcribe with Whisper ──
        print("🔄 Transcribing speech...")
        t_step = time.time()
        text, detected_lang = transcribe_audio(
            self.whisper_model, audio_data=audio_data
        )
        t_whisper = time.time() - t_step

        if not text:
            print("⚠️  No speech detected. Try again.")
            return

        src_name = LANGUAGE_NAMES.get(detected_lang, detected_lang)
        print(f"📝 [{src_name}]: \"{text}\"  (Whisper: {t_whisper:.2f}s)")

        # ── Step 2: Determine target language ──
        target_lang = AUTO_TRANSLATE_MAP.get(detected_lang)
        if target_lang is None:
            print(f"⚠️  Language '{detected_lang}' not supported. Only EN ↔ ES.")
            return

        tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)

        # ── Step 3: Translate ──
        t_step = time.time()
        translated = translate_text(
            text, self.translation_models, detected_lang, target_lang
        )
        t_translate = time.time() - t_step
        print(f"🔄 [{tgt_name}]: \"{translated}\"  (Translate: {t_translate:.2f}s)")

        # ── Step 4: Verify translation quality ──
        result = verify_translation(
            text, translated, self.translation_models,
            detected_lang, target_lang, self.embedding_model
        )
        t_verify = result['time_seconds']
        print_verification(result, time.time() - t_pipeline)

        # ── Step 5: Speak the translation & save output audio ──
        output_wav = os.path.join(RECORDINGS_DIR, f"{timestamp}_output_{target_lang}.wav")
        print(f"🔊 Speaking in {tgt_name}...")
        t_step = time.time()

        # Generate TTS to the saved file, then play it
        from translate import text_to_speech, play_audio
        tts_result = text_to_speech(translated, lang=target_lang, output_file=output_wav)
        if tts_result:
            play_audio(output_wav)
            print(f"💾 Output saved: {output_wav}")
        else:
            # Fallback: use speak() which uses a temp file
            speak(translated, lang=target_lang)

        t_tts = time.time() - t_step

        # ── Summary ──
        t_total = time.time() - t_pipeline
        print(f"\n  📊 Pipeline breakdown:")
        print(f"     Whisper (STT):    {t_whisper:.2f}s")
        print(f"     Translation:      {t_translate:.2f}s")
        print(f"     Verification:     {t_verify:.2f}s")
        print(f"     TTS + playback:   {t_tts:.2f}s")
        print(f"     ─────────────────────────")
        print(f"     Total:            {t_total:.2f}s")
        print(f"     📁 Files: {RECORDINGS_DIR}/")

    # ─────────────────────────────────────
    # MAIN LOOPS
    # ─────────────────────────────────────
    def run_gpio_mode(self):
        """Main loop using GPIO button — runs until Ctrl+C."""
        print("🟢 Ready! Hold the button (GPIO 17) to speak.\n")

        # Bind button events
        self.button.when_pressed = self.start_recording
        self.button.when_released = self.stop_recording

        # Open audio stream from USB mic at its native rate
        with sd.InputStream(samplerate=self.mic_rate,
                            channels=CHANNELS,
                            blocksize=self.block_size,
                            device=self.mic_index,
                            dtype="float32",
                            callback=self.audio_callback):
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass

    @staticmethod
    def _get_key():
        """
        Read a single keypress without waiting for Enter.
        Uses raw terminal mode so we can detect 'r' instantly.
        Returns the character pressed (lowercase).
        """
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch.lower()

    def run_keyboard_mode(self):
        """
        Keyboard mode — press 'r' to start recording, 'r' again to stop.
        Single keypress, no need to hold or press Enter.
        Press 'q' or Ctrl+C to quit.
        """
        print("🟢 Ready!")
        print("   Press [r] to start/stop recording")
        print("   Press [q] to quit\n")

        # Open audio stream from USB mic at its native rate
        with sd.InputStream(samplerate=self.mic_rate,
                            channels=CHANNELS,
                            blocksize=self.block_size,
                            device=self.mic_index,
                            dtype="float32",
                            callback=self.audio_callback):
            try:
                while True:
                    key = self._get_key()

                    # Ctrl+C sends '\x03'
                    if key in ('q', '\x03'):
                        break

                    if key == 'r':
                        if self.is_processing:
                            print("  ⏳ Still processing... wait.\n")
                            continue

                        if not self.is_recording:
                            # Start recording
                            self.start_recording()
                        else:
                            # Stop recording → triggers processing
                            self.stop_recording()

                            # Wait for processing to finish before accepting input
                            while self.is_processing:
                                time.sleep(0.1)

            except (KeyboardInterrupt, EOFError):
                pass

    def run(self):
        """Start the device."""
        print("\n" + "=" * 50)
        print("  🌐 Edge Translator Device")
        print("  Raspberry Pi 5 — USB Audio")
        print("=" * 50)

        # Configure USB audio devices
        print("\n📡 Detecting USB audio devices...")
        mic_idx, spk_idx, mic_rate = configure_audio_devices(
            self.mic_index, self.speaker_index
        )
        self.mic_index = mic_idx
        self.mic_rate = mic_rate
        # ~100ms chunks at the mic's native rate
        self.block_size = int(self.mic_rate * 0.1)

        if mic_idx is None:
            print("\n❌ No microphone found. Plug in a USB mic and try again.")
            sys.exit(1)

        # Load AI models
        self.load_models()

        # Show device summary
        print("─" * 50)
        mode = "GPIO push-to-talk" if self.use_gpio else "Keyboard [r] to toggle"
        print(f"  Mode:       {mode}")
        print(f"  Audio in:   USB mic [{mic_idx}] @ {self.mic_rate} Hz")
        print(f"  Audio out:  Speaker [{spk_idx}]")
        print(f"  Whisper in: {WHISPER_RATE} Hz {'(resampling)' if self.mic_rate != WHISPER_RATE else '(native)'}")
        print("─" * 50 + "\n")

        # Run the appropriate mode
        if self.use_gpio:
            self.run_gpio_mode()
        else:
            self.run_keyboard_mode()

        # Cleanup
        print("\n👋 Shutting down.")
        if self.led_ready:
            self.led_ready.off()
        if self.led_busy:
            self.led_busy.off()


def main():
    parser = argparse.ArgumentParser(
        description="Edge Translator Device — USB Audio + Optional GPIO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python edge_device.py                # Keyboard mode (default)
  python edge_device.py --gpio         # Use GPIO button for push-to-talk
  python edge_device.py --list-audio   # See all USB audio devices
  python edge_device.py --mic 1        # Use device [1] as microphone
  python edge_device.py --speaker 3    # Use device [3] as speaker
        """
    )
    parser.add_argument("--gpio", action="store_true",
                        help="Enable GPIO push-to-talk button and LEDs")
    parser.add_argument("--list-audio", action="store_true",
                        help="List all audio devices and exit")
    parser.add_argument("--mic", type=int, default=None,
                        help="Input device index for USB microphone")
    parser.add_argument("--speaker", type=int, default=None,
                        help="Output device index for USB speaker")
    args = parser.parse_args()

    # Just list devices and exit
    if args.list_audio:
        list_audio_devices()
        print("\nUsage: python edge_device.py --mic <INDEX> --speaker <INDEX>\n")
        sys.exit(0)

    device = TranslatorDevice(
        use_gpio=args.gpio,
        mic_index=args.mic,
        speaker_index=args.speaker,
    )
    device.run()


if __name__ == "__main__":
    main()
