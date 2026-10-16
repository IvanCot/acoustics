"""One deterministic preprocessing path for training, evaluation and prediction."""
import hashlib
import io
import os
from pathlib import Path

# Numba otherwise probes the read-only interpreter's site-packages cache on Windows.
_cache = Path(__file__).resolve().parent.parent / "artifacts" / ".numba-cache"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_cache))

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 2.0
MAX_BYTES = 12 * 1024 * 1024
PIPELINE_VERSION = "impact-cascade-v2"


class ExperimentError(ValueError):
    pass


def detection_features(y: np.ndarray) -> np.ndarray:
    """Describe the entire recording, before onset cropping or peak normalization."""
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    normalized = y / max(peak, 1e-9)
    magnitude = np.abs(librosa.stft(normalized, n_fft=512, hop_length=160))
    mel = librosa.feature.melspectrogram(S=magnitude ** 2, sr=SAMPLE_RATE, n_mels=64)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel, ref=np.max, top_db=80), n_mfcc=13)
    matrices = [mfcc, librosa.feature.spectral_centroid(S=magnitude, sr=SAMPLE_RATE),
                librosa.feature.spectral_flatness(S=magnitude),
                librosa.feature.zero_crossing_rate(normalized, frame_length=512, hop_length=160)]
    spectrum = np.concatenate([np.r_[m.mean(axis=1), m.std(axis=1)] for m in matrices])
    envelope = librosa.feature.rms(y=normalized, frame_length=512, hop_length=160)[0]
    maximum = max(float(envelope.max()), 1e-9)
    relative = envelope / maximum
    active = relative > 0.2
    starts = int(np.sum(np.diff(np.r_[False, active].astype(int)) == 1))
    temporal = [len(y) / SAMPLE_RATE, np.log10(max(peak, 1e-9)), np.log10(max(rms, 1e-9)),
                peak / max(rms, 1e-9), float(relative.mean()), float(relative.std()),
                float(np.mean(relative > 0.05)), float(active.mean()), starts,
                float(np.mean(np.maximum(np.diff(relative), 0)))]
    return np.r_[spectrum, temporal]


def process(raw: bytes, filename: str, *, allow_unsuitable: bool = False) -> dict:
    if Path(filename).suffix.lower() not in {".wav", ".mp3"}:
        raise ExperimentError("Поддерживаются только WAV и MP3.")
    if not raw or len(raw) > MAX_BYTES:
        raise ExperimentError("Пустой файл или размер больше 12 МБ.")
    try:
        with sf.SoundFile(io.BytesIO(raw)) as source:
            if source.format not in {"WAV", "WAVEX", "RF64", "MP3"}:
                raise ExperimentError("Содержимое файла не соответствует WAV или MP3.")
            if not 8000 <= source.samplerate <= 192000 or not 1 <= source.channels <= 8:
                raise ExperimentError("Нужен звук 8–192 кГц, от 1 до 8 каналов.")
            duration = source.frames / source.samplerate
            if not 0.15 <= duration <= 10:
                raise ExperimentError("Нужна аудиозапись длительностью от 0,15 до 10 секунд.")
            native_sr = source.samplerate
            samples = source.read(dtype="float32", always_2d=True)
    except ExperimentError:
        raise
    except (RuntimeError, ValueError, sf.LibsndfileError) as exc:
        raise ExperimentError("Не удалось прочитать аудио. Проверьте файл; для MP3 нужен libsndfile с поддержкой MPEG.") from exc
    if not np.isfinite(samples).all():
        raise ExperimentError("Аудио содержит некорректные числовые значения.")
    clipping = float(np.mean(np.abs(samples) >= 0.999))
    y = samples.mean(axis=1)
    y = y - np.mean(y)
    peak = float(np.max(np.abs(y)))
    rejection = None
    if peak < 0.001 or float(np.sqrt(np.mean(y ** 2))) < 0.0001:
        rejection = {"code": "silence", "reason": "Слишком тихо или тишина. Поднесите микрофон и запишите стук заново."}
    elif clipping > 0.02:
        rejection = {"code": "clipping", "reason": "Сильная перегрузка микрофона. Уменьшите громкость удара и повторите запись."}
    if rejection and not allow_unsuitable:
        raise ExperimentError(rejection["reason"])
    y = librosa.resample(y, orig_sr=native_sr, target_sr=SAMPLE_RATE)
    detector_features = detection_features(y)
    # Align on the earliest high-energy frame, not the arbitrary file start.
    energy = librosa.feature.rms(y=y, frame_length=256, hop_length=64)[0]
    hit = int(np.flatnonzero(energy >= 0.35 * energy.max())[0]) * 64
    start = max(0, hit - int(0.08 * SAMPLE_RATE))
    segment = y[start:start + int(DURATION * SAMPLE_RATE)]
    segment = librosa.util.fix_length(segment, size=int(DURATION * SAMPLE_RATE))
    if not rejection or rejection["code"] != "silence":
        segment = segment / max(float(np.max(np.abs(segment))), 1e-9) * 0.95
    segment = segment.astype(np.float32)
    magnitude = np.abs(librosa.stft(segment, n_fft=512, hop_length=160))
    power = magnitude ** 2
    mel = librosa.feature.melspectrogram(S=power, sr=SAMPLE_RATE, n_mels=64, fmax=8000)
    db = librosa.power_to_db(mel, ref=np.max, top_db=80)
    mfcc = librosa.feature.mfcc(S=db, n_mfcc=13)
    active = librosa.feature.rms(y=segment, frame_length=512, hop_length=160)[0] > 0.01
    if not active.any():
        active[:] = True  # Silence is a valid negative example for the detector.
    # Exclude the padded tail from aggregation; do not learn original file length.
    matrices = [mfcc,
                librosa.feature.spectral_centroid(S=magnitude, sr=SAMPLE_RATE),
                librosa.feature.spectral_bandwidth(S=magnitude, sr=SAMPLE_RATE),
                librosa.feature.spectral_rolloff(S=magnitude, sr=SAMPLE_RATE),
                librosa.feature.spectral_flatness(S=magnitude),
                librosa.feature.zero_crossing_rate(segment, frame_length=512, hop_length=160)]
    features = np.concatenate([np.r_[m[:, active].mean(axis=1), m[:, active].std(axis=1)] for m in matrices])
    frame_rms = librosa.feature.rms(y=segment, frame_length=512, hop_length=160)[0]
    features = np.r_[features, np.mean(frame_rms[:30]), np.mean(frame_rms[30:80]), np.mean(frame_rms[80:])]
    warnings = []
    if duration > 3:
        warnings.append("Взято 2 секунды от начала первого громкого события. Лишние удары могут исказить результат.")
    if clipping > 0.001:
        warnings.append("Есть признаки перегрузки микрофона; лучше повторить запись.")
    buckets = segment.reshape(800, 40)
    return {"features": features, "gate_features": detector_features, "rejection": rejection,
            "samples": segment, "waveform": np.stack([buckets.min(axis=1), buckets.max(axis=1)], axis=1).tolist(),
            "spectrogram": np.round(db[:, ::2], 1).tolist(),
            "canonical_hash": hashlib.sha256(np.round(segment, 4).tobytes()).hexdigest(),
            "raw_hash": hashlib.sha256(raw).hexdigest(), "warnings": warnings,
            "info": {"source_duration": round(duration, 3), "source_sample_rate": native_sr,
                     "sample_rate": SAMPLE_RATE, "duration": DURATION, "features": len(features),
                     "centroid_hz": round(float(matrices[1][:, active].mean())),
                     "onset_seconds": round(start / SAMPLE_RATE, 3)}}


def playback(samples: np.ndarray) -> bytes:
    output = io.BytesIO()
    sf.write(output, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return output.getvalue()
