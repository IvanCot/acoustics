"""Convert RealImpact's deconvolved IronPlate array into labelled PCM WAV files."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "preprocessed"
OUTPUT = HERE / "wav"
SAMPLE_RATE = 48_000  # RealImpact's modes_dsp_sweep.ipynb


def main() -> None:
    audio = np.load(SOURCE / "deconvolved_0db.npy", mmap_mode="r", allow_pickle=False)
    angle = np.load(SOURCE / "angle.npy", mmap_mode="r", allow_pickle=False)
    distance = np.load(SOURCE / "distance.npy", mmap_mode="r", allow_pickle=False)
    mic_id = np.load(SOURCE / "micID.npy", mmap_mode="r", allow_pickle=False)
    vertex_id = np.load(SOURCE / "vertexID.npy", mmap_mode="r", allow_pickle=False)

    if audio.ndim != 2 or any(len(x) != len(audio) for x in (angle, distance, mic_id, vertex_id)):
        raise ValueError("Audio and annotation arrays have inconsistent dimensions")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {OUTPUT}")
    OUTPUT.mkdir(exist_ok=True)

    # Apply one shared scale so relative levels between recordings remain meaningful.
    global_peak = 0.0
    for row in range(len(audio)):
        global_peak = max(global_peak, float(np.max(np.abs(audio[row]))))
    if not np.isfinite(global_peak) or global_peak == 0:
        raise ValueError("Audio array has no finite nonzero signal")
    scale = 0.95 / global_peak

    with (OUTPUT / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("filename", "row", "vertex_id", "angle_deg", "distance_mm", "mic_id", "sample_rate_hz", "samples"))
        for row in range(len(audio)):
            values = np.asarray(audio[row], dtype=np.float32)
            pcm = np.rint(values * scale * 32767).clip(-32768, 32767).astype(np.int16)
            name = (
                f"ironplate_v{int(vertex_id[row])}_angle{int(angle[row]):03d}"
                f"_distance{int(distance[row]):04d}_mic{int(mic_id[row]):02d}_row{row:04d}.wav"
            )
            wavfile.write(OUTPUT / name, SAMPLE_RATE, pcm)
            writer.writerow((name, row, int(vertex_id[row]), int(angle[row]), int(distance[row]), int(mic_id[row]), SAMPLE_RATE, len(pcm)))
            if (row + 1) % 250 == 0:
                print(f"Converted {row + 1}/{len(audio)}")

    (OUTPUT / "README.txt").write_text(
        "RealImpact IronPlate conversion\n"
        f"Files: {len(audio)} mono PCM16 WAV recordings\n"
        f"Sample rate: {SAMPLE_RATE} Hz (from RealImpact modes_dsp_sweep.ipynb)\n"
        f"Duration per recording: {audio.shape[1] / SAMPLE_RATE:.3f} seconds\n"
        "Source: preprocessed/deconvolved_0db.npy\n"
        "A single global gain scale was applied to preserve relative levels and avoid clipping.\n"
        "metadata.csv maps each WAV to its original row and RealImpact annotations.\n",
        encoding="utf-8",
    )
    print(f"Done: {len(audio)} WAV files, {audio.shape[1] / SAMPLE_RATE:.3f} s each, global peak {global_peak:.6g}")


if __name__ == "__main__":
    main()
