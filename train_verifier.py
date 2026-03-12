#!/usr/bin/env python3
"""Train a custom verifier model for the igor wake word.

The verifier is a speaker-specific second stage (scikit-learn LogisticRegression)
that runs after the base ONNX model fires. It replaces the base score with a
speaker-verified probability, cutting ~95% of false positives.

Prerequisites:
  - Trained base model at oww_models/igor.onnx
  - Positive samples in wakeword_samples/positive/*.wav (16kHz mono)
  - Negative samples in wakeword_samples/negative/*.wav (16kHz mono)
  - pip install scikit-learn (if not already installed)

Output:
  oww_models/igor_verifier.pkl

Deploy to Pi:
  scp oww_models/igor_verifier.pkl pi@<PI_IP>:~/smart_assistant/oww_models/

Must retrain if you retrain the base ONNX model (embedding features change).
"""

import glob
import sys
from pathlib import Path

ONNX_MODEL = "oww_models/igor.onnx"
OUTPUT_PATH = "oww_models/igor_verifier.pkl"

def main():
    if not Path(ONNX_MODEL).exists():
        print(f"Base model not found: {ONNX_MODEL}")
        print("Train the base model first: python onnx_models/wakeword_creation/train_wakeword.py")
        sys.exit(1)

    positive_clips = sorted(glob.glob("wakeword_samples/positive/*.wav"))
    negative_clips = sorted(glob.glob("wakeword_samples/negative/*.wav"))

    if not positive_clips:
        print("No positive samples found in wakeword_samples/positive/")
        sys.exit(1)
    if not negative_clips:
        print("No negative samples found in wakeword_samples/negative/")
        print("Record TV/music/speech audio into wakeword_samples/negative/ first.")
        sys.exit(1)

    print(f"Positive clips: {len(positive_clips)}")
    print(f"Negative clips: {len(negative_clips)}")
    print(f"Base model: {ONNX_MODEL}")
    print()

    try:
        import openwakeword
    except ImportError:
        print("openwakeword not installed: pip install openwakeword")
        sys.exit(1)

    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("scikit-learn not installed: pip install scikit-learn")
        sys.exit(1)

    print("Training verifier (this may take a minute)...")
    # OWW's train_custom_verifier has a Windows bug: it splits the model path
    # with os.path.sep but forward slashes survive splitext, producing
    # "oww_models/igor" instead of "igor". Work around by passing the absolute
    # path (which uses backslashes on Windows).
    abs_model = str(Path(ONNX_MODEL).resolve())
    openwakeword.train_custom_verifier(
        positive_reference_clips=positive_clips,
        negative_reference_clips=negative_clips,
        output_path=OUTPUT_PATH,
        model_name=abs_model,
        inference_framework="onnx",
    )

    size_kb = Path(OUTPUT_PATH).stat().st_size / 1024
    print(f"\nVerifier saved: {OUTPUT_PATH} ({size_kb:.1f} KB)")
    print(f"\nDeploy to Pi:")
    print(f"  scp {OUTPUT_PATH} pi@<PI_IP>:~/smart_assistant/oww_models/")
    print(f"\nThe client auto-detects *_verifier.pkl files in oww_models/.")
    print(f"You may need to lower OWW_THRESHOLD from 0.75 to ~0.5-0.6")
    print(f"since verifier scores are more conservative than raw base scores.")


if __name__ == "__main__":
    main()
