"""Convert metal_manual_v2 and replace the metal recordings in train/val/test."""

import csv
import json
import random
import shutil
import subprocess
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE = DATA / "metal_manual_v2"
SESSION = "manual_20261017_metal_v2"
SPLITS = ("train", "val", "test")
FFMPEG = Path(r"C:\Users\20072\AppData\Local\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe")
FIELDS = ("split", "class", "session_id", "new_filename", "source_filename", "source_group", "source_bytes", "wav_bytes")


def main():
    sources = sorted(p for p in SOURCE.iterdir() if p.is_file() and p.suffix.lower() in {".aac", ".m4a", ".wav"})
    if not sources:
        raise SystemExit(f"В {SOURCE} не найдены аудиозаписи.")
    if not FFMPEG.is_file():
        raise SystemExit(f"Не найден ffmpeg: {FFMPEG}")

    rng = random.Random(20261017)
    order = list(range(len(sources)))
    rng.shuffle(order)
    n_train = int(len(order) * 0.70)
    n_val = (len(order) - n_train) // 2
    split_indices = {
        "train": set(order[:n_train]),
        "val": set(order[n_train:n_train + n_val]),
        "test": set(order[n_train + n_val:]),
    }

    manifest_path = DATA / "manual_split_manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as f:
        old_manifest = list(csv.DictReader(f))
    wood_rows = [r for r in old_manifest if r["class"] != "metal"]
    new_rows = []

    stage = SOURCE / "converted_wav"
    stage.mkdir(exist_ok=True)
    for old in stage.glob("metal_v2_*.wav"):
        old.unlink()
    try:
        staged_files = {}
        for i, source in enumerate(sources):
            target = stage / f"metal_v2_{i + 1:02d}.wav"
            subprocess.run([
                str(FFMPEG), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(source), "-map_metadata", "-1", "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(target),
            ], check=True)
            with wave.open(str(target), "rb") as audio:
                if audio.getnframes() == 0 or audio.getnchannels() != 1 or audio.getframerate() != 16000:
                    raise RuntimeError(f"Неверный WAV после конвертации: {source.name}")
            staged_files[i] = target

        for split in SPLITS:
            for i in sorted(split_indices[split]):
                source = sources[i]
                filename = f"metal_v2_{i + 1:02d}.wav"
                new_rows.append({
                    "split": split, "class": "metal", "session_id": SESSION,
                    "new_filename": filename, "source_filename": source.name,
                    "source_group": SESSION, "source_bytes": source.stat().st_size,
                    "wav_bytes": staged_files[i].stat().st_size,
                })

        # Prepare metadata before replacing any existing recordings.
        combined = wood_rows + new_rows
        manifest_tmp = stage / "manual_split_manifest.csv"
        with manifest_tmp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(combined)
        policy_path = DATA / "split_policy.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        policy["file_split_sessions"] = sorted(set(policy["file_split_sessions"] + [SESSION]))
        policy_tmp = stage / "split_policy.json"
        policy_tmp.write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        for split in SPLITS:
            metal_root = DATA / split / "metal"
            if metal_root.exists():
                for old in metal_root.rglob("*"):
                    if old.is_file() and old.suffix.lower() in {".wav", ".mp3", ".m4a", ".aac"}:
                        old.unlink()
        for i, staged in staged_files.items():
            split = next(name for name, indexes in split_indices.items() if i in indexes)
            destination = DATA / split / "metal" / SESSION / staged.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staged), destination)

        shutil.move(str(manifest_tmp), manifest_path)
        shutil.move(str(policy_tmp), policy_path)
    finally:
        for old in stage.glob("metal_v2_*.wav"):
            old.unlink(missing_ok=True)

    counts = {split: sum(r["split"] == split for r in new_rows) for split in SPLITS}
    wood_counts = {split: sum(r["split"] == split for r in wood_rows) for split in SPLITS}
    (DATA / "MANUAL_SPLIT_README.txt").write_text(
        "Ручные записи: дерево из исходной сессии и металл из metal_manual_v2; все WAV — моно PCM16, 16 кГц.\n"
        f"Металл ({SESSION}): {counts}; дерево: {wood_counts}. Разделение новых записей детерминировано seed 20261017, приблизительно 70/15/15.\n"
        "Исходные аудиофайлы сохранены. Сопоставление исходников с WAV: manual_split_manifest.csv.\n"
        "Условия записи выборок могут зависеть; метрики по этому split не являются независимой оценкой обобщения.\n",
        encoding="utf-8",
    )
    print(f"Конвертировано: {len(sources)}; металл по split: {counts}; дерево сохранено: {wood_counts}")


if __name__ == "__main__":
    main()
