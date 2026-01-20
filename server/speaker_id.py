"""Speaker identification using Resemblyzer voice embeddings."""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Lazy load resemblyzer to avoid import errors if not installed
_encoder = None


def _get_encoder():
    """Lazy load the voice encoder."""
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder()
        logger.info("Loaded Resemblyzer voice encoder")
    return _encoder


@dataclass
class SpeakerMatch:
    """Result of speaker identification."""
    name: str
    confidence: float
    is_known: bool


class SpeakerIdentifier:
    """Identify speakers using voice embeddings."""

    def __init__(self, embeddings_file: Path, similarity_threshold: float = 0.75):
        """
        Initialize speaker identifier.

        Args:
            embeddings_file: Path to JSON file storing speaker embeddings
            similarity_threshold: Minimum cosine similarity to consider a match (0-1)
        """
        self.embeddings_file = embeddings_file
        self.similarity_threshold = similarity_threshold
        self.speakers: dict[str, np.ndarray] = {}
        self._load_embeddings()

    def _load_embeddings(self):
        """Load speaker embeddings from file."""
        if self.embeddings_file.exists():
            try:
                with open(self.embeddings_file, 'r') as f:
                    data = json.load(f)
                for name, embedding_list in data.items():
                    self.speakers[name] = np.array(embedding_list)
                logger.info(f"Loaded {len(self.speakers)} speaker embeddings: {list(self.speakers.keys())}")
            except Exception as e:
                logger.error(f"Failed to load speaker embeddings: {e}")
                self.speakers = {}
        else:
            logger.info("No speaker embeddings file found, starting fresh")

    def _save_embeddings(self):
        """Save speaker embeddings to file."""
        try:
            self.embeddings_file.parent.mkdir(parents=True, exist_ok=True)
            data = {name: emb.tolist() for name, emb in self.speakers.items()}
            with open(self.embeddings_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.speakers)} speaker embeddings")
        except Exception as e:
            logger.error(f"Failed to save speaker embeddings: {e}")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def enroll_speaker(self, name: str, audio_samples: list[np.ndarray], sample_rate: int = 16000) -> bool:
        """
        Enroll a new speaker with voice samples.

        Args:
            name: Speaker's name
            audio_samples: List of audio arrays (numpy float32, mono)
            sample_rate: Sample rate of audio (default 16000)

        Returns:
            True if enrollment succeeded
        """
        try:
            from resemblyzer import preprocess_wav
            encoder = _get_encoder()

            embeddings = []
            for audio in audio_samples:
                # Preprocess audio
                wav = preprocess_wav(audio, source_sr=sample_rate)
                # Get embedding
                embedding = encoder.embed_utterance(wav)
                embeddings.append(embedding)

            # Average embeddings for more robust representation
            avg_embedding = np.mean(embeddings, axis=0)
            self.speakers[name] = avg_embedding
            self._save_embeddings()

            logger.info(f"Enrolled speaker '{name}' with {len(audio_samples)} samples")
            return True

        except Exception as e:
            logger.error(f"Failed to enroll speaker '{name}': {e}")
            return False

    def enroll_from_file(self, name: str, audio_path: Path) -> bool:
        """
        Enroll a speaker from an audio file.

        Args:
            name: Speaker's name
            audio_path: Path to audio file (WAV recommended)

        Returns:
            True if enrollment succeeded
        """
        try:
            from resemblyzer import preprocess_wav
            import soundfile as sf

            # Load audio file
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert to mono

            return self.enroll_speaker(name, [audio], sample_rate=sr)

        except Exception as e:
            logger.error(f"Failed to enroll from file '{audio_path}': {e}")
            return False

    def identify(self, audio: np.ndarray, sample_rate: int = 16000) -> SpeakerMatch:
        """
        Identify the speaker from audio.

        Args:
            audio: Audio array (numpy float32, mono)
            sample_rate: Sample rate of audio

        Returns:
            SpeakerMatch with name, confidence, and whether speaker is known
        """
        if not self.speakers:
            return SpeakerMatch(name="unknown", confidence=0.0, is_known=False)

        try:
            from resemblyzer import preprocess_wav
            encoder = _get_encoder()

            # Preprocess and get embedding
            wav = preprocess_wav(audio, source_sr=sample_rate)
            embedding = encoder.embed_utterance(wav)

            # Find best match
            best_name = "unknown"
            best_score = 0.0

            for name, speaker_embedding in self.speakers.items():
                similarity = self._cosine_similarity(embedding, speaker_embedding)
                if similarity > best_score:
                    best_score = similarity
                    best_name = name

            is_known = best_score >= self.similarity_threshold

            if is_known:
                logger.debug(f"Identified speaker as '{best_name}' (confidence: {best_score:.2f})")
            else:
                logger.debug(f"Unknown speaker (best match: '{best_name}' at {best_score:.2f})")

            return SpeakerMatch(
                name=best_name if is_known else "unknown",
                confidence=best_score,
                is_known=is_known
            )

        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
            return SpeakerMatch(name="unknown", confidence=0.0, is_known=False)

    def remove_speaker(self, name: str) -> bool:
        """Remove a speaker from the database."""
        if name in self.speakers:
            del self.speakers[name]
            self._save_embeddings()
            logger.info(f"Removed speaker '{name}'")
            return True
        return False

    def list_speakers(self) -> list[str]:
        """List all enrolled speakers."""
        return list(self.speakers.keys())


# Convenience function for quick identification
def identify_speaker(audio: np.ndarray, embeddings_file: Path, sample_rate: int = 16000) -> SpeakerMatch:
    """
    Quick speaker identification without persistent instance.

    Args:
        audio: Audio array
        embeddings_file: Path to embeddings file
        sample_rate: Sample rate

    Returns:
        SpeakerMatch result
    """
    identifier = SpeakerIdentifier(embeddings_file)
    return identifier.identify(audio, sample_rate)
