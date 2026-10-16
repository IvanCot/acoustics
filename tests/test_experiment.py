"""Engineering checks on synthetic signals, never evidence of material accuracy."""
import base64
import io
import json
import shutil
import subprocess
import threading
import unittest
import uuid
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf

from acoustic.audio import ExperimentError, light_denoise, process, training_waveform
from acoustic.experiment import Experiment
from server import make_handler

TEST_ROOT = Path(__file__).resolve().parent.parent / "test-output"


class temporary_workspace:
    """Keep inherited workspace ACLs; Python's private temp DACL blocks sandbox tokens."""
    def __init__(self):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        root = TEST_ROOT / ("run-" + uuid.uuid4().hex)
        root.mkdir()
        self.name = str(root)

    def cleanup(self):
        root = Path(self.name).resolve()
        assert root.is_relative_to(TEST_ROOT.resolve()) and root != TEST_ROOT.resolve()
        shutil.rmtree(root)


def signal(seed=0, frequency=500, rate=22050, duration=2.5, channels=1):
    """Damped tone plus noise; not a recording of any real material."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(rate * duration)) / rate
    relative = np.maximum(t - .4, 0)
    y = .5 * np.sin(2*np.pi*frequency*relative) * np.exp(-relative * 7)
    y[t < .4] = 0
    y += rng.normal(0, .0003, y.shape)
    if channels == 2:
        y = np.stack([y, y * .8], axis=1)
    output = io.BytesIO()
    sf.write(output, y, rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def background(seed=0, silent=False):
    """Synthetic continuous noise; checks mechanics, not speech rejection accuracy."""
    y = np.zeros(40000) if silent else np.random.default_rng(seed).normal(0, .05, 40000)
    output = io.BytesIO()
    sf.write(output, y, 16000, format="WAV", subtype="PCM_16")
    return output.getvalue()


class AudioTests(unittest.TestCase):
    def test_light_denoiser_reduces_background_and_preserves_impact(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, .004, 32000).astype(np.float32)
        y[8000] += .7
        clean = light_denoise(y)
        self.assertEqual(clean.shape, y.shape)
        self.assertTrue(np.isfinite(clean).all())
        self.assertLess(np.mean(clean[:6000] ** 2), np.mean(y[:6000] ** 2))
        self.assertGreater(abs(clean[8000]), .95 * abs(y[8000]))
        np.testing.assert_array_equal(light_denoise(np.zeros(32000)), np.zeros(32000))

    def test_augmentation_is_repeatable_and_used_before_shared_denoiser(self):
        y = np.sin(np.arange(16000) * .1).astype(np.float32) * .1
        a = training_waveform(y, 'a' * 64, 0)
        np.testing.assert_array_equal(a, training_waveform(y, 'a' * 64, 0))
        self.assertFalse(np.allclose(a, training_waveform(y, 'a' * 64, 1)))
        self.assertEqual(a.shape, y.shape)
        with self.assertRaises(ValueError):
            training_waveform(y, 'a' * 64, 2)
        raw = signal()
        first = process(raw, 'sample.wav', training_variant=0)
        second = process(raw, 'sample.wav', training_variant=0)
        np.testing.assert_array_equal(first['features'], second['features'])
        self.assertEqual(first['raw_hash'], process(raw, 'sample.wav')['raw_hash'])

    def test_one_pipeline_resamples_stereo_aligns_and_pads(self):
        result = process(signal(rate=44100, channels=2), "stereo.wav")
        self.assertEqual(result["samples"].shape, (32000,))
        self.assertEqual(result["features"].shape, (39,))
        self.assertTrue(np.isfinite(result["features"]).all())
        self.assertAlmostEqual(float(np.max(np.abs(result["samples"]))), .95, places=5)
        self.assertGreater(result["info"]["onset_seconds"], .2)
        self.assertEqual(len(result["spectrogram"]), 64)

    def test_rejects_silence_corrupt_wrong_type_and_long_audio(self):
        silent = io.BytesIO()
        sf.write(silent, np.zeros(16000), 16000, format="WAV")
        for raw, name in [(silent.getvalue(), "quiet.wav"), (b"not audio", "broken.wav"),
                          (signal(), "run.py"), (signal(duration=11), "long.wav"),
                          (signal(duration=.1), "short.wav")]:
            with self.subTest(name=name), self.assertRaises(ExperimentError):
                process(raw, name)

    def test_mp3_decoding(self):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("MP3 encoder fixture needs ffmpeg; application decoding doesn't.")
        result = subprocess.run([ffmpeg, "-v", "error", "-f", "wav", "-i", "pipe:0", "-f", "mp3", "pipe:1"],
                                input=signal(), capture_output=True, check=True)
        audio = process(result.stdout, "fixture.mp3")
        self.assertEqual(len(audio["samples"]), 32000)
        self.assertTrue(np.isfinite(audio["features"]).all())

    def test_detector_uses_entire_recording_and_silence_has_finite_features(self):
        raw, rate = sf.read(io.BytesIO(signal(duration=5)))
        changed = raw.copy()
        changed[rate * 3:] += np.random.default_rng(9).normal(0, .02, changed[rate * 3:].shape)
        output = io.BytesIO()
        sf.write(output, changed, rate, format="WAV", subtype="PCM_16")
        first = process(signal(duration=5), "first.wav")
        second = process(output.getvalue(), "second.wav")
        # The suffix changes the full-file DC offset slightly; the material window stays the same.
        np.testing.assert_allclose(first["samples"], second["samples"], atol=.001)
        self.assertFalse(np.allclose(first["gate_features"], second["gate_features"]))
        silence = process(background(silent=True), "silence.wav", allow_unsuitable=True)
        self.assertEqual(silence["rejection"]["code"], "silence")
        self.assertTrue(np.isfinite(silence["gate_features"]).all())
        self.assertTrue(np.isfinite(silence["features"]).all())


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_workspace()
        self.root = Path(self.temp.name)
        self.experiment = Experiment(self.root / "data", self.root / "artifacts")

    def tearDown(self):
        assert self.root.resolve().is_relative_to(TEST_ROOT.resolve())
        self.temp.cleanup()

    def populate(self):
        for label, frequency in [("wood", 400), ("metal", 2400)]:
            for i in range(4):
                self.experiment.add(signal(i, frequency+i*29), "fixture.wav", "train", label, "s01")
            for i in range(2):
                self.experiment.add(signal(20+i, frequency+i*43+11), "fixture.wav", "test", label, "s02")
        for i in range(4):
            self.experiment.add(background(i), "background.wav", "train", "other", "s01")
        for i in range(2):
            self.experiment.add(background(20+i), "background.wav", "test", "other", "s02")

    def test_empty_is_untrained_and_blocks_train_and_evaluation(self):
        state = self.experiment.state()
        self.assertFalse(state["trained"])
        self.assertIsNone(state["report"])
        result = self.experiment.analyze(signal(), "new.wav")
        self.assertIsNone(result["prediction"])
        self.assertIsNone(result["scores"])
        with self.assertRaises(ExperimentError): self.experiment.train()
        with self.assertRaises(ExperimentError): self.experiment.evaluate()

    def test_duplicates_sessions_and_path_traversal(self):
        raw = signal()
        self.experiment.add(raw, "../../outside.wav", "train", "wood", "s01")
        self.assertFalse((self.root / "outside.wav").exists())
        with self.assertRaises(ExperimentError):
            self.experiment.add(raw, "duplicate.wav", "test", "metal", "s02")
        with self.assertRaises(ExperimentError):
            self.experiment.add(signal(6), "new.wav", "test", "wood", "s01")
        with self.assertRaises(ExperimentError):
            self.experiment.add(signal(7), "new.wav", "train", "wood", "../../bad")
        decoded, rate = sf.read(io.BytesIO(raw))
        reencoded = io.BytesIO()
        sf.write(reencoded, decoded, rate, format="WAV", subtype="FLOAT")
        with self.assertRaises(ExperimentError):
            self.experiment.add(reencoded.getvalue(), "same.wav", "train", "metal", "s01")

    def test_full_cycle_persistence_report_and_staleness(self):
        self.populate()
        trained = self.experiment.train()
        self.assertTrue(trained["trained"])
        report = self.experiment.evaluate()
        self.assertEqual(report["n_test"], 6)
        self.assertEqual(sum(sum(row) for row in report["confusion"]), 6)
        self.assertEqual(report["labels"], ["metal", "wood", "other"])
        self.assertEqual(report["gate"]["n_other"], 2)
        self.assertEqual(report["gate"]["n_knocks"], 4)
        self.assertEqual(report["sessions"], ["s02"])
        result = self.experiment.analyze(signal(99, 700), "new.wav", "material")
        self.assertIn(result["prediction"], ["wood", "metal"])
        self.assertAlmostEqual(sum(result["scores"].values()), 1)
        reloaded = Experiment(self.root / "data", self.root / "artifacts")
        self.assertEqual(result["scores"], reloaded.analyze(signal(99, 700), "new.wav", "material")["scores"])
        self.experiment.add(signal(98, 800), "extra.wav", "train", "wood", "s01")
        self.assertTrue(self.experiment.state()["stale"])
        self.assertIsNone(self.experiment.state()["report"])
        self.assertIsNone(self.experiment.analyze(signal(99), "new.wav")["prediction"])

    def test_explicit_file_split_warns_keeps_val_out_of_fit_and_rejects_duplicates(self):
        self.populate()
        raw = signal(100, 850)
        with self.assertRaises(ExperimentError):
            self.experiment.add(raw, "validation.wav", "val", "wood", "s01")
        (self.root / "data" / "split_policy.json").write_text(
            json.dumps({"file_split_sessions": ["s01"]}), encoding="utf-8")
        self.experiment.add(raw, "validation.wav", "val", "wood", "s01")
        state = self.experiment.train("material")
        self.assertTrue(state["material_trained"])
        self.assertEqual(state["model"]["n_train"], 8)
        self.assertTrue(state["warnings"])
        self.assertFalse(state["issues"])
        with self.assertRaises(ExperimentError):
            self.experiment.add(raw, "duplicate.wav", "train", "wood", "s01")
        self.assertTrue(self.experiment.evaluate("material")["warnings"])

    def test_gate_blocks_material_scores_and_direct_mode_bypasses_gate(self):
        self.populate()
        self.experiment.train()
        raw = signal(99)
        with patch.object(self.experiment, "scores", return_value={"knock": .2, "other": .8}) as scores:
            result = self.experiment.analyze(raw, "new.wav")
            self.assertEqual(result["gate"]["status"], "rejected")
            self.assertIsNone(result["prediction"])
            self.assertIsNone(result["scores"])
            self.assertEqual(scores.call_count, 1)
        with patch.object(self.experiment, "scores", side_effect=[{"knock": .8, "other": .2}, {"wood": .7, "metal": .3}]) as scores:
            result = self.experiment.analyze(raw, "new.wav")
            self.assertEqual(result["prediction"], "wood")
            self.assertEqual(scores.call_count, 2)
        with patch.object(self.experiment, "scores", return_value={"wood": .7, "metal": .3}) as scores:
            result = self.experiment.analyze(raw, "new.wav", "material")
            self.assertEqual(result["gate"]["status"], "skipped")
            self.assertEqual(result["prediction"], "wood")
            self.assertEqual(scores.call_count, 1)
            clipped = io.BytesIO()
            sf.write(clipped, np.tile([1., -1.], 20000), 16000, format="WAV", subtype="PCM_16")
            for mode in ["cascade", "material"]:
                for raw, code in [(background(silent=True), "silence"), (clipped.getvalue(), "clipping")]:
                    scores.reset_mock()
                    result = self.experiment.analyze(raw, "invalid.wav", mode)
                    self.assertEqual(result["gate"]["code"], code)
                    self.assertIsNone(result["prediction"])
                    scores.assert_not_called()
        with self.assertRaises(ExperimentError):
            self.experiment.analyze(raw, "new.wav", "invalid")

    def test_models_train_independently_and_negative_changes_only_stale_gate(self):
        self.populate()
        self.experiment.train("material")
        state = self.experiment.state()
        self.assertTrue(state["material_trained"])
        self.assertFalse(state["gate_trained"])
        self.assertIsNone(self.experiment.analyze(signal(70), "new.wav")["prediction"])
        self.assertIsNotNone(self.experiment.analyze(signal(70), "new.wav", "material")["prediction"])
        report = self.experiment.evaluate("material")
        self.assertEqual(report["n_test"], 4)
        self.assertNotIn("gate", report)
        self.experiment.train("gate")
        self.assertTrue(self.experiment.state()["trained"])
        original = self.experiment.read_json("model.json")
        self.experiment.add(background(90), "noise.wav", "train", "other", "s01")
        state = self.experiment.state()
        self.assertFalse(state["gate_trained"])
        self.assertTrue(state["material_trained"])
        self.assertEqual(state["report"]["mode"], "material")
        self.experiment.train("gate")
        self.assertEqual(original, self.experiment.read_json("model.json"))

    def test_all_rejected_report_counts_true_knocks_as_errors(self):
        self.populate()
        self.experiment.train()
        with patch.object(self.experiment, "scores", return_value={"knock": .1, "other": .9}):
            report = self.experiment.evaluate()
        self.assertAlmostEqual(report["accuracy"], 2 / 6)
        self.assertEqual(report["gate"]["false_rejects"], 4)
        self.assertEqual(report["gate"]["false_accepts"], 0)
        self.assertIsNone(report["material_accepted"]["accuracy"])
        self.assertEqual(report["material_accepted"]["coverage"], 0)

    def test_silence_can_only_be_added_as_negative_and_missing_negative_blocks_gate(self):
        raw = background(silent=True)
        with self.assertRaises(ExperimentError):
            self.experiment.add(raw, "silence.wav", "train", "wood", "s01")
        self.experiment.add(raw, "silence.wav", "train", "other", "s01")
        self.assertEqual(self.experiment.state()["counts"]["train"]["other"], 1)
        with self.assertRaises(ExperimentError): self.experiment.train("gate")

    def test_test_change_cannot_change_fitted_scaler_or_coefficients(self):
        self.populate()
        self.experiment.train()
        first = self.experiment.read_json("model.json")
        first_gate = self.experiment.read_json("gate.json")
        self.assertEqual(first["n_fit"], 3 * first["n_train"])
        self.assertEqual(first_gate["n_fit"], 3 * first_gate["n_train"])
        self.experiment.evaluate()
        self.experiment.add(signal(60, 6000), "outlier.wav", "test", "wood", "s02")
        self.experiment.add(signal(61, 5900), "validation.wav", "val", "wood", "s03")
        self.assertIsNone(self.experiment.state()["report"])
        self.assertTrue(self.experiment.state()["trained"])
        self.experiment.train()
        second = self.experiment.read_json("model.json")
        second_gate = self.experiment.read_json("gate.json")
        for key in ["mean", "scale", "coef", "intercept"]:
            self.assertEqual(first[key], second[key])
            self.assertEqual(first_gate[key], second_gate[key])

    def test_duplicate_copied_on_disk_blocks_training(self):
        self.populate()
        rows, _ = self.experiment.scan()
        original = self.experiment.data / rows[0]["path"]
        target = self.experiment.data / "test" / "wood" / "s02" / "copied.wav"
        target.write_bytes(original.read_bytes())
        self.assertFalse(self.experiment.state()["can_train"])
        self.assertTrue(self.experiment.state()["issues"])


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = temporary_workspace()
        root = Path(cls.temp.name)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Experiment(root / "data", root / "artifacts")))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()
        assert Path(cls.temp.name).resolve().is_relative_to(TEST_ROOT.resolve())
        cls.temp.cleanup()

    def test_static_state_audio_and_validation(self):
        with urlopen(self.url) as response:
            self.assertIn("Акустический детектив", response.read().decode())
            self.assertIn("script-src 'self'", response.headers["Content-Security-Policy"])
        with urlopen(self.url + "/api/state") as response:
            self.assertFalse(json.load(response)["trained"])
        payload = {"filename": "test.wav", "audio": base64.b64encode(signal()).decode()}
        request = Request(self.url + "/api/analyze", json.dumps(payload).encode(), {"Content-Type": "application/json"})
        with urlopen(request) as response:
            result = json.load(response)
            self.assertEqual(result["info"]["features"], 39)
            self.assertIsNone(result["prediction"])
            self.assertEqual(result["mode"], "cascade")
            self.assertEqual(result["gate"]["status"], "not_trained")
        payload["mode"] = "material"
        request = Request(self.url + "/api/analyze", json.dumps(payload).encode(), {"Content-Type": "application/json"})
        with urlopen(request) as response:
            self.assertEqual(json.load(response)["gate"]["status"], "skipped")
        bad = Request(self.url + "/api/analyze", b'{"audio":"!"}', {"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as context: urlopen(bad)
        self.assertEqual(context.exception.code, 400)

    def test_cross_site_host_and_unknown_paths_rejected(self):
        for request, code in [
            (Request(self.url + "/api/train", b"{}", {"Content-Type": "application/json", "Origin": "https://evil.example"}), 403),
            (Request(self.url + "/api/state", headers={"Host": "evil.example"}), 403),
            (Request(self.url + "/../server.py"), 404),
            (Request(self.url + "/api/train", b"{}", {"Content-Type": "text/plain"}), 400),
            (Request(self.url + "/api/train", b"{}", {"Content-Type": "application/json", "Content-Length": "999999999"}), 413)
        ]:
            with self.subTest(code=code), self.assertRaises(HTTPError) as context: urlopen(request)
            self.assertEqual(context.exception.code, code)


if __name__ == "__main__":
    unittest.main()
