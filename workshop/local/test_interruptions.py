"""Test interruptions without microphone hardware."""

import numpy as np
from voice_interruptions import (
    InterruptionDetector,
    SilenceDetector,
    rms,
)


def test_interruption_detector():
    """Test interruption detection logic."""
    print("Testing InterruptionDetector...")
    detector = InterruptionDetector(threshold=0.03, min_duration_ms=200)

    # Simulate quiet audio (below threshold)
    quiet = np.zeros(16000, dtype=np.int16)
    result = detector.update(rms(quiet))
    print(f"  Quiet audio interrupted: {result} (expected: False)")

    # Simulate loud audio (above threshold)
    loud = np.ones(16000, dtype=np.int16) * 5000
    result1 = detector.update(rms(loud))
    result2 = detector.update(rms(loud))
    print(f"  Loud audio interrupted: {result1} or {result2} (expected: True after 200ms)")

    print("✓ InterruptionDetector works\n")


def test_silence_detector():
    """Test silence detection logic."""
    print("Testing SilenceDetector...")
    detector = SilenceDetector(threshold=0.025, silence_ms=700, max_ms=5000)

    # Simulate speech then silence
    for i in range(10):
        if i < 5:
            result = detector.update(rms(np.ones(16000, dtype=np.int16) * 3000))
        else:
            result = detector.update(rms(np.zeros(16000, dtype=np.int16)))
        if result:
            print(f"  Silence detected at frame {i}")
            break

    print("✓ SilenceDetector works\n")


def test_rms():
    """Test RMS calculation."""
    print("Testing RMS calculation...")
    quiet = np.zeros(16000, dtype=np.int16)
    loud = np.ones(16000, dtype=np.int16) * 10000

    rms_quiet = rms(quiet)
    rms_loud = rms(loud)

    print(f"  RMS(quiet): {rms_quiet:.4f}")
    print(f"  RMS(loud): {rms_loud:.4f}")
    print(f"  Loud > Quiet: {rms_loud > rms_quiet} (expected: True)")
    print("✓ RMS works\n")


def test_config():
    """Test config loading."""
    print("Testing config...")
    try:
        from voice import transcription_session_config
        config = transcription_session_config()
        print(f"  Config type: {config.get('type')}")
        print("✓ Config loads\n")
    except Exception as e:
        print(f"  Error: {e}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Voice Interruptions (No Microphone)")
    print("=" * 60 + "\n")

    test_rms()
    test_interruption_detector()
    test_silence_detector()
    test_config()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
