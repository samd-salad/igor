#!/usr/bin/env python3
"""Train an OpenWakeWord model from recorded positive samples.

Run this on your PC after recording samples with record_samples.py on the Pi.

Workflow:
  1. Pi:  python record_samples.py           → wakeword_samples/positive/*.wav
  2. Pi→PC: rsync wakeword_samples/ to PC
  3. PC:  python onnx_models/wakeword_creation/train_wakeword.py
  4. PC→Pi: copy oww_models/doctor_butts.onnx to Pi's oww_models/

Requirements (PC only):
  pip install openwakeword[training]
"""
import sys
from pathlib import Path

# Project root is two levels up from this script
ROOT       = Path(__file__).parent.parent.parent
POSITIVE_DIR = ROOT / "wakeword_samples" / "positive"
OUTPUT_DIR   = ROOT / "oww_models"
MODEL_NAME   = "doctor_butts"


def main():
    wav_files = sorted(POSITIVE_DIR.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {POSITIVE_DIR}/")
        print("Record samples first: python record_samples.py  (on the Pi)")
        sys.exit(1)

    print(f"Found {len(wav_files)} positive samples.")
    if len(wav_files) < 50:
        print(f"Warning: {len(wav_files)} samples is low — aim for 100+ for reliable detection.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"{MODEL_NAME}.onnx"

    try:
        from openwakeword.train import train_model
    except ImportError:
        print("\nTraining dependencies not installed.")
        print("Run: pip install openwakeword[training]")
        sys.exit(1)

    print(f"\nTraining '{MODEL_NAME}' model...")
    print("First run downloads base models (~50 MB). Subsequent runs are faster.\n")

    train_model(
        model_name=MODEL_NAME,
        positive_clips_dir=str(POSITIVE_DIR),
        output_dir=str(OUTPUT_DIR),
    )

    if output_path.exists():
        size_kb = output_path.stat().st_size / 1024
        print(f"\nModel saved: {output_path}  ({size_kb:.0f} KB)")
        print(f"\nNext: copy {output_path.name} to the Pi's oww_models/ directory")
        print(f"  scp {output_path} pi@<PI_IP>:~/smart_assistant/oww_models/")
    else:
        # Some versions write to a subdirectory — show what was created
        onnx_files = list(OUTPUT_DIR.rglob("*.onnx"))
        if onnx_files:
            print(f"\nModel files found in {OUTPUT_DIR}/:")
            for f in onnx_files:
                print(f"  {f}")
        else:
            print(f"\nTraining complete. Check {OUTPUT_DIR}/ for output files.")


if __name__ == "__main__":
    main()
