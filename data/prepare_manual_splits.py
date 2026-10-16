"""Convert manual AAC/M4A captures to WAV and make deterministic 70/15/15 splits."""

from __future__ import annotations

import csv
import json
import random
import subprocess
from pathlib import Path


DATA = Path(__file__).resolve().parent
FFMPEG = Path(r"C:\Users\20072\AppData\Local\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe")
SEED = 20261016
SPLITS = ("train", "val", "test")
SESSION = "manual_20261016"
SOURCES = {"metal": DATA / "metal_manual", "wood": DATA / "wood_manual"}


def allocation(n: int) -> dict[str, int]:
    train = int(n * 0.70)
    remainder = n - train
    val = remainder // 2
    return {"train": train, "val": val, "test": remainder - val}


def main() -> None:
    if not FFMPEG.is_file():
        raise FileNotFoundError(f"ffmpeg not found: {FFMPEG}")
    destinations = [DATA / split / label / SESSION for split in SPLITS for label in SOURCES]
    existing = [p for p in destinations if p.exists() and any(p.iterdir())]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing split data: {existing}")

    manifest: list[dict[str, object]] = []
    for label, source_dir in SOURCES.items():
        files = sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in {".m4a", ".aac"})
        if not files:
            raise FileNotFoundError(f"No .m4a/.aac recordings in {source_dir}")
        order = list(range(len(files)))
        random.Random(SEED + (1 if label == "metal" else 2)).shuffle(order)
        amounts = allocation(len(files))
        slices: dict[str, list[int]] = {}
        offset = 0
        for split in SPLITS:
            end = offset + amounts[split]
            slices[split] = order[offset:end]
            offset = end

        for split in SPLITS:
            for original_index in sorted(slices[split]):
                source = files[original_index]
                destination = DATA / split / label / SESSION / f"{label}{original_index + 1}.wav"
                destination.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    str(FFMPEG), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(source), "-map_metadata", "-1", "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(destination),
                ]
                subprocess.run(command, check=True)
                manifest.append(
                    {
                        "split": split,
                        "class": label,
                        "session_id": SESSION,
                        "new_filename": destination.name,
                        "source_filename": source.name,
                        "source_group": source.stem.rsplit(" ", 1)[0] if label == "metal" else "wood_manual",
                        "source_bytes": source.stat().st_size,
                        "wav_bytes": destination.stat().st_size,
                    }
                )

    manifest_path = DATA / "manual_split_manifest.csv"
    policy_path = DATA / "split_policy.json"
    policy = json.loads(policy_path.read_text("utf-8")) if policy_path.exists() else {}
    policy["file_split_sessions"] = sorted(set(policy.get("file_split_sessions", []) + [SESSION]))
    policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    counts = {split: {label: sum(r["split"] == split and r["class"] == label for r in manifest)
                      for label in SOURCES} for split in SPLITS}
    (DATA / "MANUAL_SPLIT_README.txt").write_text(
        "Ручные записи: AAC/M4A сконвертированы в моно PCM16 WAV 16 кГц.\n"
        "Распределение детерминировано seed 20261016, приблизительно 70/15/15 по классам.\n"
        f"Число файлов: {counts}.\n"
        "Исходники оставлены в metal_manual и wood_manual. Имена новых файлов связаны с исходниками в manual_split_manifest.csv.\n"
        f"Исходные файлы не размечены по независимым сессиям; всем присвоен общий ID ({SESSION}), при явном разрешении разбиения по файлам через split_policy.json. Приложение показывает предупреждение о зависимости выборок, а дубликаты продолжает блокировать. Метрики по такому split нельзя считать независимым holdout.\n"
        "Для честной финальной оценки запишите отдельную новую сессию и держите её целиком в test.\n",
        encoding="utf-8",
    )
    print("Counts:", counts)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
