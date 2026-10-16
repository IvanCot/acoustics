"""Build a small app-compatible WAV subset from the ReaLISED filename labels."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "selected_by_material"
MATERIALS = {"01": "wood", "03": "metal", "04": "plastic"}
TARGET_RATE = 16_000
MIN_SECONDS = 0.15
MAX_SECONDS = 10.0


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {OUTPUT}")
    for label in MATERIALS.values():
        (OUTPUT / label).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    skipped_short: list[str] = []
    for source in sorted(ROOT.glob("*.flac")):
        parts = source.stem.split("_")
        if len(parts) != 5:
            continue
        event, event_id, action, material_code, intensity = parts
        label = MATERIALS.get(material_code)
        if label is None:
            continue

        audio, sample_rate = sf.read(source, dtype="float32", always_2d=False)
        if audio.ndim != 1:
            raise ValueError(f"Expected mono source, got {audio.shape}: {source.name}")
        duration = len(audio) / sample_rate
        if duration < MIN_SECONDS:
            skipped_short.append(source.name)
            continue
        if duration > MAX_SECONDS:
            raise ValueError(f"Clip exceeds the app's 10 s limit: {source.name} ({duration:.3f} s)")
        if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1.00001:
            raise ValueError(f"Invalid float audio range: {source.name}")

        if sample_rate != TARGET_RATE:
            audio = resample_poly(audio, TARGET_RATE, sample_rate).astype(np.float32, copy=False)
        pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        destination = OUTPUT / label / f"{source.stem}.wav"
        wavfile.write(destination, TARGET_RATE, pcm)
        manifest.append(
            {
                "wav": str(destination.relative_to(OUTPUT)).replace("\\", "/"),
                "source_flac": source.name,
                "material": label,
                "material_code": material_code,
                "event": event,
                "event_id": event_id,
                "action_code": action,
                "intensity_code": intensity,
                "source_sample_rate_hz": sample_rate,
                "duration_seconds": f"{duration:.4f}",
            }
        )

    with (OUTPUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    counts = {label: sum(1 for row in manifest if row["material"] == label) for label in MATERIALS.values()}
    readme = [
        "ReaLISED material subset",
        f"Selected clips: {len(manifest)}; output: mono PCM16 WAV at {TARGET_RATE} Hz.",
        "Material codes in the source filename: 01=wood, 03=metal, 04=plastic.",
        "Counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()),
        f"Skipped clips shorter than {MIN_SECONDS:.2f} s: {len(skipped_short)}.",
        "Original FLACs are unchanged. See manifest.csv for source labels and event/action codes.",
        "Dataset: ReaLISED, Mohino-Herranz et al. (2022), CC BY 4.0.",
        "https://doi.org/10.5281/zenodo.6488321",
    ]
    (OUTPUT / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"Selected {len(manifest)} clips; skipped {len(skipped_short)} too-short clip(s): {skipped_short}")
    print("Counts:", counts)


if __name__ == "__main__":
    main()
