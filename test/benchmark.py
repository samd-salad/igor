"""Performance benchmarking and logging for voice assistant pipeline."""

import csv
import time
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

BENCHMARK_FILE = Path(__file__).parent / "benchmark.csv"


class Benchmark:
    """Tracks timing for STT, LLM, and TTS stages with CSV logging."""

    def __init__(self, csv_path: Path = BENCHMARK_FILE):
        self.csv_path = csv_path
        self._ensure_csv()
        self._cache: dict[tuple[str, str], list[float]] = {}
        self._load_cache()

    def _ensure_csv(self):
        """Create CSV with headers if it doesn't exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'stage', 'model', 'duration_s', 'word_count', 'per_word_s'])

    def _load_cache(self):
        """Load existing data into memory for average calculations."""
        if not self.csv_path.exists():
            return
        try:
            with open(self.csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row['stage'], row['model'])
                    if key not in self._cache:
                        self._cache[key] = []
                    # For STT, track per-word time; for others, track raw duration
                    if row['stage'] == 'stt' and row['per_word_s']:
                        self._cache[key].append(float(row['per_word_s']))
                    elif row['duration_s']:
                        self._cache[key].append(float(row['duration_s']))
        except Exception:
            pass

    def log(self, stage: str, model: str, duration: float, word_count: Optional[int] = None):
        """Log a timing entry to CSV and update cache."""
        per_word = duration / word_count if word_count and word_count > 0 else None

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S'),
                stage,
                model,
                f'{duration:.3f}',
                word_count if word_count else '',
                f'{per_word:.3f}' if per_word else ''
            ])

        # Update cache
        key = (stage, model)
        if key not in self._cache:
            self._cache[key] = []
        if stage == 'stt' and per_word:
            self._cache[key].append(per_word)
        else:
            self._cache[key].append(duration)

    def get_average(self, stage: str, model: str) -> Optional[float]:
        """Get average timing for a stage/model combination."""
        key = (stage, model)
        values = self._cache.get(key, [])
        if not values:
            return None
        return sum(values) / len(values)

    def get_count(self, stage: str, model: str) -> int:
        """Get number of samples for a stage/model combination."""
        key = (stage, model)
        return len(self._cache.get(key, []))

    def format_stt_log(self, duration: float, word_count: int, model: str) -> str:
        """Format STT timing message with average comparison."""
        per_word = duration / word_count if word_count > 0 else 0
        msg = f"{duration:.2f}s to convert speech to text"
        if word_count > 0:
            msg += f", or {per_word:.2f}s per word"

        avg = self.get_average('stt', model)
        if avg:
            comparison = "Below" if per_word < avg else "Above"
            msg += f". {comparison} {avg:.2f}s average for {model} whisper"

        return msg

    def format_llm_log(self, duration: float, model: str) -> str:
        """Format LLM timing message with average comparison."""
        msg = f"{duration:.2f}s for LLM response"

        avg = self.get_average('llm', model)
        if avg:
            comparison = "Below" if duration < avg else "Above"
            msg += f". {comparison} {avg:.2f}s average for {model}"

        return msg

    def format_tts_log(self, duration: float, model: str, word_count: int = 0) -> str:
        """Format TTS timing message with average comparison."""
        msg = f"{duration:.2f}s for text-to-speech"
        if word_count > 0:
            per_word = duration / word_count
            msg += f" ({per_word:.2f}s per word)"

        avg = self.get_average('tts', model)
        if avg:
            comparison = "Below" if duration < avg else "Above"
            msg += f". {comparison} {avg:.2f}s average for {model}"

        return msg

    @contextmanager
    def time_stage(self, stage: str, model: str, word_count: Optional[int] = None):
        """Context manager for timing a stage. Word count can be set after via .word_count attribute."""
        class TimingContext:
            def __init__(ctx):
                ctx.word_count = word_count
                ctx.duration = 0.0

        ctx = TimingContext()
        start = time.perf_counter()
        try:
            yield ctx
        finally:
            ctx.duration = time.perf_counter() - start
            self.log(stage, model, ctx.duration, ctx.word_count)


# Global instance
_benchmark: Optional[Benchmark] = None


def get_benchmark() -> Benchmark:
    """Get or create the global benchmark instance."""
    global _benchmark
    if _benchmark is None:
        _benchmark = Benchmark()
    return _benchmark
