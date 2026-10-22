"""Local voice pipeline with interruption support. Run: python voice_interruptions.py --help

Features: record with interruption detection, STT, LLM, TTS with barge-in handling.
"""
import argparse
import os
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).with_name(".env"))
RATE = 16000
FRAME_MS = 30
CHANNELS = 1
SAMPLE_DTYPE = "int16"


def save_wav(path, samples, rate=RATE):
    samples = np.asarray(samples, dtype=np.int16).reshape(-1)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())


def rms(samples):
    x = np.asarray(samples, dtype=np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


class SilenceDetector:
    def __init__(self, threshold=0.025, silence_ms=700, max_ms=15000):
        self.threshold, self.silence_ms, self.max_ms = threshold, silence_ms, max_ms
        self.elapsed_ms = 0
        self.last_speech_ms = None

    def update(self, volume, frame_ms=FRAME_MS):
        self.elapsed_ms += frame_ms
        if volume > self.threshold:
            self.last_speech_ms = self.elapsed_ms
        if self.elapsed_ms >= self.max_ms:
            return True
        return (self.last_speech_ms is not None and
                self.elapsed_ms - self.last_speech_ms >= self.silence_ms)


class InterruptionDetector:
    """Detect user speech while assistant is speaking (ignores noise spikes)."""
    def __init__(self, threshold=0.06, min_duration_ms=500, noise_floor=0.02):
        self.threshold = threshold  # Speech volume threshold
        self.min_duration_ms = min_duration_ms  # Require sustained speech
        self.noise_floor = noise_floor  # Minimum audio to consider
        self.speech_start_ms = None
        self.interrupted = False
        self.recent_volumes = []

    def is_speech(self, volume):
        """Check if audio is likely speech (not noise spike)."""
        # Must be above noise floor
        if volume < self.noise_floor:
            return False
        # Track recent volumes for stability check
        self.recent_volumes.append(volume)
        if len(self.recent_volumes) > 5:
            self.recent_volumes.pop(0)
        # Require consistent loudness (not a spike)
        avg_vol = np.mean(self.recent_volumes) if self.recent_volumes else 0
        return volume > self.threshold and avg_vol > self.noise_floor

    def update(self, volume, frame_ms=FRAME_MS):
        if self.is_speech(volume):
            if self.speech_start_ms is None:
                self.speech_start_ms = 0
            self.speech_start_ms += frame_ms
            if self.speech_start_ms >= self.min_duration_ms:
                self.interrupted = True
                return True
        else:
            self.speech_start_ms = None
        return False


def record_until_silence(threshold=0.025, silence_ms=700, max_seconds=15):
    detector = SilenceDetector(threshold, silence_ms, int(max_seconds * 1000))
    frames = []
    print("Speak naturally. Stop speaking to finish...")
    blocksize = RATE * FRAME_MS // 1000
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                        blocksize=blocksize) as stream:
        while True:
            frame, overflowed = stream.read(blocksize)
            if overflowed:
                print("Warning: microphone input overflow")
            frames.append(frame.copy())
            if detector.update(rms(frame)):
                break
    if detector.last_speech_ms is None:
        raise RuntimeError("No sound detected. Try a lower --threshold.")
    return np.concatenate(frames).reshape(-1)


def record_with_interruption_detection(max_seconds=15, threshold=0.03):
    """Record user speech while detecting interruptions."""
    detector = InterruptionDetector(threshold=threshold)
    frames = []
    print("Recording user speech. Speak to interrupt...")
    print(f"Listening for {max_seconds} seconds (threshold: {threshold})...")
    blocksize = RATE * FRAME_MS // 1000
    try:
        with sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                            blocksize=blocksize) as stream:
            start_time = time.time()
            while time.time() - start_time < max_seconds:
                frame, overflowed = stream.read(blocksize)
                if overflowed:
                    print(".", end="", flush=True)
                frames.append(frame.copy())
                if detector.update(rms(frame)):
                    print("\n✓ Interruption detected!")
                    break
        print("Recording complete.")
    except Exception as e:
        print(f"Error: {e}")
        raise
    if not frames:
        raise RuntimeError("No audio recorded")
    return np.concatenate(frames).reshape(-1)


def transcribe(client, path):
    with open(path, "rb") as audio:
        result = client.audio.transcriptions.create(
            model=os.getenv("VOICE_STT_MODEL", "gpt-4o-mini-transcribe"),
            file=audio,
        )
    return result.text.strip()


def think(client, text, fast=False):
    model = "gpt-4o-mini" if fast else os.getenv("VOICE_LLM_MODEL", "gpt-4.1-mini")
    result = client.responses.create(
        model=model,
        instructions="You are a concise, helpful shopping assistant. Reply in 1-2 sentences.",
        input=text,
    )
    return result.output_text.strip()


def speak_with_interruption(client, text, detect_interrupts=True):
    """Play TTS while listening for interruptions (full duplex)."""
    interrupt_detected = threading.Event()

    def listen_for_interrupt():
        detector = InterruptionDetector(threshold=0.05, min_duration_ms=500)
        blocksize = RATE * FRAME_MS // 1000
        try:
            with sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                               blocksize=blocksize) as stream:
                for _ in range(int(15 * 1000 / FRAME_MS)):  # Listen for 15 seconds max
                    frame, _ = stream.read(blocksize)
                    if detector.update(rms(frame)):
                        interrupt_detected.set()
                        break
        except Exception as e:
            print(f"Error during interrupt detection: {e}")

    print("Playing AI-generated speech. Speak to interrupt..." if detect_interrupts else "Playing response...")
    start = time.perf_counter()
    first_audio_ms = None
    underflows = 0
    pending = b""

    if detect_interrupts:
        listener_thread = threading.Thread(target=listen_for_interrupt, daemon=True)
        listener_thread.start()

    try:
        with client.audio.speech.with_streaming_response.create(
            model=os.getenv("VOICE_TTS_MODEL", "gpt-4o-mini-tts"),
            voice=os.getenv("VOICE_TTS_VOICE", "marin"),
            input=text,
            response_format="pcm",
        ) as response:
            with sd.RawOutputStream(samplerate=24000, channels=1, dtype="int16") as stream:
                for chunk in response.iter_bytes(chunk_size=960):
                    if interrupt_detected.is_set():
                        print("✓ Interruption detected! Stopping playback.")
                        stream.abort()
                        return True
                    pending += chunk
                    complete = len(pending) - len(pending) % 2
                    if not complete:
                        continue
                    audio = np.frombuffer(pending[:complete], dtype="<i2").astype(
                        np.int16, copy=False).tobytes()
                    pending = pending[complete:]
                    if first_audio_ms is None:
                        first_audio_ms = (time.perf_counter() - start) * 1000
                        print(f"First audio submitted: {first_audio_ms:.0f} ms")
                    if stream.write(audio):
                        underflows += 1
                if pending:
                    raise RuntimeError("TTS returned an incomplete 16-bit PCM sample.")
    except Exception as e:
        print(f"TTS error: {e}")
        return False

    elapsed = (time.perf_counter() - start) * 1000
    print(f"TTS playback: {elapsed:.0f} ms")
    if underflows:
        print(f"Warning: {underflows} speaker underflow(s)")
    return False


def cascade_with_interruptions(client, samples, fast=False):
    """Full pipeline with interruption handling."""
    save_wav("recording.wav", samples)
    start = time.perf_counter()
    text = transcribe(client, "recording.wav")
    stt_ms = round((time.perf_counter() - start) * 1000)
    print(f"You: {text} (STT {stt_ms} ms)")
    if not text:
        return False

    start = time.perf_counter()
    answer = think(client, text, fast=fast)
    llm_ms = round((time.perf_counter() - start) * 1000)
    print(f"Assistant: {answer} (LLM {llm_ms} ms)")

    interrupted = speak_with_interruption(client, answer)
    return interrupted


def main():
    parser = argparse.ArgumentParser(
        description="Voice pipeline with interruption support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python voice_interruptions.py cascade          # Full pipeline
  python voice_interruptions.py loop             # Continuous conversation
  python voice_interruptions.py detect           # Test interruption detection
  python voice_interruptions.py full-duplex      # TTS with interrupt listening
        """,
    )
    parser.add_argument("stage", choices=[
        "cascade", "loop", "detect", "full-duplex", "stt", "llm", "tts",
    ])
    parser.add_argument("--threshold", type=float, default=0.025)
    parser.add_argument("--silence-ms", type=int, default=700)
    parser.add_argument("--text", default="Hello! Welcome to AxiomCart.")
    parser.add_argument("--interrupt-threshold", type=float, default=0.03)
    parser.add_argument("--fast", action="store_true", help="Use faster (gpt-4o-mini) LLM")
    parser.add_argument("--no-interrupt", action="store_true", help="Disable interruption detection (test mode)")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in workshop/local/.env")
    client = OpenAI()

    if args.stage == "detect":
        print("Testing interruption detection...")
        samples = record_with_interruption_detection()
        save_wav("interruption_test.wav", samples)
        print("Saved interruption_test.wav")

    elif args.stage == "full-duplex":
        print("Testing full duplex (TTS with interruption)...")
        speak_with_interruption(client, args.text)

    elif args.stage == "stt":
        samples = record_until_silence(args.threshold, args.silence_ms)
        save_wav("recording.wav", samples)
        print(transcribe(client, "recording.wav"))

    elif args.stage == "llm":
        print(think(client, args.text))

    elif args.stage == "tts":
        speak_with_interruption(client, args.text, detect_interrupts=not args.no_interrupt)

    elif args.stage == "cascade":
        samples = record_until_silence(args.threshold, args.silence_ms)
        cascade_with_interruptions(client, samples, fast=args.fast)

    elif args.stage == "loop":
        print("Ctrl+C to stop. Full interruption handling enabled.")
        try:
            while True:
                samples = record_until_silence(args.threshold, args.silence_ms)
                interrupted = cascade_with_interruptions(client, samples, fast=args.fast)
                if interrupted:
                    print("\nWaiting for next turn...")
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
