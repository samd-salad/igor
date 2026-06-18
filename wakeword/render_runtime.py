"""Render systemd ExecStart for wyoming-openwakeword using values from contracts.py.

Print to stdout when run as a script; importable for use in tests / deploy scripts.
"""
from __future__ import annotations
import sys

from wakeword import contracts


def render_openwakeword_execstart(
    run_script: str,
    custom_model_dir: str,
    model_name: str,
    uri: str = "tcp://0.0.0.0:10400",
    threshold: float = contracts.DEFAULT_THRESHOLD,
    trigger_level: int = contracts.DEFAULT_TRIGGER_LEVEL,
) -> str:
    return (
        f"{run_script} "
        f"--uri {uri} "
        f"--custom-model-dir {custom_model_dir} "
        f"--preload-model {model_name} "
        f"--threshold {threshold} "
        f"--trigger-level {trigger_level} "
        f"--debug"
    )


def main() -> int:
    line = render_openwakeword_execstart(
        run_script="/home/samda/wyoming-openwakeword/script/run",
        custom_model_dir="/home/samda/wyoming-openwakeword/custom-models",
        model_name="igor",
    )
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
