"""Select duration-matched background recordings for the knock detector."""

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import soundfile as sf

DATA = Path(__file__).resolve().parent
SOURCE = DATA.parent / "ReaLISED_Dataset"
FFMPEG = Path(r"C:\Users\20072\AppData\Local\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe")
SESSION = "realised_background"
CATEGORIES = {"spe": "speech", "tel": "television", "wat": "water", "vac": "vacuum_cleaner"}
SPLITS = ("train", "val", "test")


def main():
    for split in SPLITS:
        dest = DATA / split / "other" / SESSION
        if dest.exists() and any(dest.iterdir()):
            raise FileExistsError(f"Already populated: {dest}")

    # Consecutive source IDs stay within large blocks rather than being shuffled
    # between splits. Original capture session/speaker IDs are unavailable.
    pools = {split: {} for split in SPLITS}
    for code in CATEGORIES:
        eligible = [(p, sf.info(str(p)).duration) for p in sorted(SOURCE.glob(f"{code}_*.flac"))]
        eligible = [(p, seconds) for p, seconds in eligible if 1.1 <= seconds <= 3.5]
        first, second = int(len(eligible) * .70), int(len(eligible) * .85)
        for split, part in zip(SPLITS, (eligible[:first], eligible[first:second], eligible[second:])):
            pools[split][code] = list(part)

    plan, seen = [], set()
    for split in SPLITS:
        positives = sorted(p for label in ("wood", "metal")
                           for p in (DATA / split / label).glob("*/*.wav"))
        for i, positive in enumerate(positives):
            code = tuple(CATEGORIES)[i % len(CATEGORIES)]
            target_duration = sf.info(str(positive)).duration
            candidates = pools[split][code]
            if not candidates:
                raise ValueError(f"Insufficient sources in {split}/{code}")
            source, duration = min(candidates, key=lambda item: (abs(item[1] - target_duration), item[0].name))
            candidates.remove((source, duration))
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest in seen:
                raise ValueError(f"Duplicate source audio: {source}")
            seen.add(digest)
            plan.append((split, code, source, duration, target_duration, digest))

    manifest = []
    for split, code, source, duration, target_duration, digest in plan:
        destination = DATA / split / "other" / SESSION / (source.stem + ".wav")
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(FFMPEG), "-nostdin", "-hide_banner", "-loglevel", "error",
                        "-n", "-i", str(source), "-map_metadata", "-1", "-vn", "-ac", "1",
                        "-ar", "16000", "-c:a", "pcm_s16le", str(destination)], check=True)
        manifest.append({"source": source.name, "source_sha256": digest,
                         "destination": destination.relative_to(DATA).as_posix(),
                         "split": split, "category": CATEGORIES[code],
                         "duration_seconds": round(duration, 4),
                         "matched_positive_duration_seconds": round(target_duration, 4)})

    policy_path = DATA / "split_policy.json"
    policy = json.loads(policy_path.read_text("utf-8")) if policy_path.exists() else {}
    policy["file_split_sessions"] = sorted(set(policy.get("file_split_sessions", []) + [SESSION]))
    policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    with (DATA / "realised_negative_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (DATA / "REALISED_NEGATIVES_README.txt").write_text(
        "ReaLISED detector negatives: speech, television, water, vacuum cleaner.\n"
        "Source labels were used; each clip has not been manually auditioned for incidental impacts.\n"
        "Counts match manual positive recordings per split. Duration range 1.1-3.5 s; nearest duration selection.\n"
        "No synthetic audio, repetition, padding or gain normalization; mono PCM16 WAV 16 kHz.\n"
        "Consecutive source ID blocks: first 70% train, next 15% val, remainder test.\n"
        "Capture-session and speaker IDs are unavailable. Splits are NOT proven independent.\n"
        "Manual positive recordings and ReaLISED negatives have different microphones/environments; source shortcuts may inflate validation results.\n"
        "ReaLISED: Mohino-Herranz et al. (2022), CC BY 4.0, https://doi.org/10.5281/zenodo.6488321\n",
        encoding="utf-8")
    print("Selected:", dict(Counter(row["split"] for row in manifest)))
    print("Categories:", dict(Counter(row["category"] for row in manifest)))


if __name__ == "__main__":
    main()
