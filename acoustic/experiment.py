"""Filesystem dataset, leakage checks, fixed baseline and honest held-out reports."""
import base64
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from .audio import ExperimentError, PIPELINE_VERSION, TRAINING_AUGMENTATION, process, playback, split_impacts

CLASSES = ["metal", "wood"]  # sklearn sorted binary class order
DATA_CLASSES = [*CLASSES, "other"]
LABELS = {"metal": "Металл", "wood": "Дерево", "other": "Неподходящий звук"}
GATE_THRESHOLD = 0.65  # Fixed before test; an uncalibrated decision score, not a guarantee.
SPLITS = ["train", "val", "test", "demo"]


def stamp():
    return datetime.now(timezone.utc).isoformat()


class Experiment:
    def __init__(self, data: Path, artifacts: Path):
        self.data, self.artifacts = Path(data), Path(artifacts)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            for label in DATA_CLASSES:
                (self.data / split / label).mkdir(parents=True, exist_ok=True)
        self.cache = {}

    def read_json(self, name):
        path = self.artifacts / name
        return json.loads(path.read_text("utf-8")) if path.exists() else None

    def write_json(self, name, value):
        path = self.artifacts / name
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
        temporary.replace(path)

    def scan(self):
        rows, errors = [], []
        for split in SPLITS:
            for label in DATA_CLASSES:
                for path in sorted((self.data / split / label).rglob("*")):
                    if not path.is_file() or path.suffix.lower() not in {".wav", ".mp3"}:
                        continue
                    relative = path.relative_to(self.data)
                    try:
                        if path.is_symlink() or not path.resolve().is_relative_to(self.data.resolve()):
                            raise ExperimentError("Ссылки за пределы папки данных запрещены.")
                        if len(relative.parts) != 4 or not re.fullmatch(r"[a-z0-9_-]{1,40}", relative.parts[2]):
                            raise ExperimentError("Нужна структура split/class/session/file.wav; session — a-z, 0-9, _ или -.")
                        if path.stat().st_size > 12 * 1024 * 1024:
                            raise ExperimentError("Файл больше 12 МБ.")
                        raw = path.read_bytes()
                        digest = hashlib.sha256(raw).hexdigest()
                        if digest not in self.cache:
                            self.cache[digest] = process(raw, path.name, allow_unsuitable=True)
                        audio = self.cache[digest]
                        if label != "other" and audio["rejection"]:
                            raise ExperimentError(audio["rejection"]["reason"])
                        rows.append({"path": relative.as_posix(), "split": split, "label": label,
                                     "session": relative.parts[2], "audio": audio})
                    except (ExperimentError, OSError) as exc:
                        errors.append({"path": relative.as_posix(), "error": str(exc)})
        # Cache only current files, so a long-running session doesn't grow forever.
        self.cache = {row["audio"]["raw_hash"]: row["audio"] for row in rows}
        return rows, errors

    @staticmethod
    def signature(rows, split, labels=None):
        content = [(r["path"], r["audio"]["raw_hash"]) for r in rows
                   if r["split"] == split and (labels is None or r["label"] in labels)]
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def integrity(rows, file_split_sessions=()):
        issues, sessions, hashes = [], {}, {}
        for row in rows:
            session = row["session"]
            if session in sessions and sessions[session] != row["split"] and session not in file_split_sessions:
                issues.append(f"Сессия {session} встречается в разных выборках. Разделяйте целые сессии.")
            sessions[session] = row["split"]
            for kind in ["raw_hash", "canonical_hash"]:
                key = kind, row["audio"][kind]
                if key in hashes and hashes[key] != row["path"]:
                    issues.append(f"Дубликат аудио: {hashes[key]} и {row['path']}. Оставьте один экземпляр.")
                hashes[key] = row["path"]
        return sorted(set(issues))

    def file_split_sessions(self):
        """Explicit file-level partitions keep provenance; duplicates still block training."""
        path = self.data / "split_policy.json"
        if not path.exists():
            return []
        sessions = json.loads(path.read_text("utf-8")).get("file_split_sessions", [])
        if not isinstance(sessions, list) or any(not isinstance(s, str) or not re.fullmatch(r"[a-z0-9_-]{1,40}", s) for s in sessions):
            raise ExperimentError("Некорректные file_split_sessions в split_policy.json.")
        return sorted(set(sessions))

    def state(self):
        rows, errors = self.scan()
        model, gate = self.read_json("model.json"), self.read_json("gate.json")
        report = self.read_json("report.json")
        validation = self.read_json("validation.json")
        if not validation or validation.get("pipeline") != PIPELINE_VERSION:
            validation = None
        counts = {s: {c: sum(r["split"] == s and r["label"] == c for r in rows) for c in DATA_CLASSES} for s in SPLITS}
        file_splits = self.file_split_sessions()
        issues = self.integrity(rows, file_splits)
        warnings = [f"Сессия {session}: разбиение по файлам; train, val и test не являются независимыми сессиями."
                    for session in file_splits if len({r["split"] for r in rows if r["session"] == session}) > 1]
        material_ready = bool(model and model.get("pipeline") == PIPELINE_VERSION and
                              model["train_signature"] == self.signature(rows, "train", CLASSES) and not errors and not issues)
        gate_ready = bool(gate and gate.get("pipeline") == PIPELINE_VERSION and
                          gate["train_signature"] == self.signature(rows, "train") and not errors and not issues)
        ready = material_ready and gate_ready
        report_mode = report.get("mode") if report else None
        report_valid = bool(material_ready and report and report_mode in {"cascade", "material"} and
                            report["model_id"] == model["id"] and
                            (report_mode == "material" or (gate_ready and report.get("gate_id") == gate["id"])) and
                            report["test_signature"] == self.signature(rows, "test", CLASSES if report_mode == "material" else None))
        can_material = all(counts["train"][c] >= 4 for c in CLASSES) and not errors and not issues
        can_gate = (counts["train"]["other"] >= 4 and
                    sum(counts["train"][c] for c in CLASSES) >= 4 and not errors and not issues)
        can_evaluate_material = material_ready and all(counts["test"][c] >= 2 for c in CLASSES)
        can_evaluate = ready and can_evaluate_material and counts["test"]["other"] >= 2
        def passport(value):
            return {k: value.get(k) for k in ["id", "created", "n_train", "n_fit", "augmentation", "sessions", "pipeline"]} if value else None
        return {"classes": LABELS, "counts": counts, "rows": [{k: v for k, v in r.items() if k != "audio"} for r in rows],
                "errors": errors, "issues": issues, "warnings": warnings, "trained": ready, "stale": bool(model and not ready),
                "material_trained": material_ready, "gate_trained": gate_ready,
                "can_train": can_material and can_gate, "can_train_material": can_material, "can_train_gate": can_gate,
                "can_evaluate": can_evaluate, "can_evaluate_material": can_evaluate_material,
                "model": passport(model), "gate_model": passport(gate),
                "report": report if report_valid else None, "validation": validation}

    def add(self, raw, filename, split, label, session):
        if split not in SPLITS or label not in DATA_CLASSES:
            raise ExperimentError("Неизвестный класс или выборка.")
        if not isinstance(session, str) or not re.fullmatch(r"[a-z0-9_-]{1,40}", session):
            raise ExperimentError("ID сессии: 1–40 символов a-z, 0-9, _ или -.")
        audio = process(raw, filename, allow_unsuitable=label == "other")
        rows, _ = self.scan()
        candidate = {"path": "новая запись", "split": split, "label": label, "session": session, "audio": audio}
        issues = self.integrity(rows + [candidate], self.file_split_sessions())
        if issues:
            raise ExperimentError(" ".join(issues))
        folder = self.data / split / label / session
        folder.mkdir(parents=True, exist_ok=True)
        # Never use a user supplied path as a destination.
        path = folder / (audio["raw_hash"][:16] + Path(filename).suffix.lower())
        path.write_bytes(raw)
        return {"saved": path.relative_to(self.data).as_posix()}

    def train(self, stage="both", *, augment=True):
        state = self.state()
        if stage not in {"both", "gate", "material"}:
            raise ExperimentError("Неизвестный этап обучения.")
        if stage in {"both", "material"} and not state["can_train_material"]:
            raise ExperimentError("Для модели материала нужно минимум по 4 разных записи дерева и металла в train без ошибок данных.")
        if stage in {"both", "gate"} and not state["can_train_gate"]:
            raise ExperimentError("Для детектора нужны минимум 4 стука (дерево/металл) и 4 неподходящих звука в train без ошибок данных.")
        rows, _ = self.scan()
        train = [r for r in rows if r["split"] == "train"]
        if augment:
            for row in train:
                raw = (self.data / row["path"]).read_bytes()
                row["augmented_audio"] = [process(raw, row["path"], allow_unsuitable=True,
                                                  training_variant=i)
                                          for i in range(TRAINING_AUGMENTATION["variants_per_recording"])]
        if stage in {"both", "material"}:
            material = [r for r in train if r["label"] in CLASSES]
            self.fit_model(material, "features", [r["label"] for r in material], "model.json", CLASSES)
        if stage in {"both", "gate"}:
            self.fit_model(train, "gate_features", ["other" if r["label"] == "other" else "knock" for r in train], "gate.json")
        return self.state()

    def fit_model(self, train, features, y, filename, labels=None):
        samples = [(audio[features], label) for row, label in zip(train, y)
                   for audio in [row["audio"], *row.get("augmented_audio", [])]]
        x = np.array([values for values, _ in samples])
        y = [label for _, label in samples]
        scaler = StandardScaler().fit(x)  # Never fit on held-out data.
        classifier = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42).fit(scaler.transform(x), y)
        created = stamp()
        signature = self.signature(train, "train", labels)
        model = {"id": hashlib.sha256((created + signature).encode()).hexdigest()[:12],
                 "pipeline": PIPELINE_VERSION, "created": created, "n_train": len(train),
                 "n_fit": len(samples),
                 "augmentation": TRAINING_AUGMENTATION if len(samples) > len(train) else None,
                 "sessions": sorted(set(r["session"] for r in train)), "train_signature": signature,
                 "file_split_sessions": self.file_split_sessions(),
                 "classes": classifier.classes_.tolist(), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                 "coef": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0])}
        if filename == "gate.json":
            model["threshold"] = GATE_THRESHOLD
        self.write_json(filename, model)  # JSON only; no untrusted pickle deserialization.

    @staticmethod
    def scores(features, model):
        x = (np.asarray(features) - model["mean"]) / model["scale"]
        logit = float(np.dot(x, model["coef"]) + model["intercept"])
        p = float(1 / (1 + np.exp(-np.clip(logit, -40, 40))))
        return {model["classes"][0]: 1 - p, model["classes"][1]: p}

    @staticmethod
    def validate_mode(mode):
        if mode not in {"cascade", "material"}:
            raise ExperimentError("Режим анализа: cascade или material.")

    def decide(self, audio, state, mode):
        gate = {"status": "skipped", "accepted": None, "scores": None, "threshold": GATE_THRESHOLD,
                "reason": "Детектор пропущен: выбран прямой прогноз материала.", "model_id": None}
        if audio["rejection"]:
            gate.update(status="rejected", accepted=False, **audio["rejection"])
        elif mode == "cascade":
            if not state["gate_trained"]:
                gate.update(status="not_trained", code="gate_untrained", reason="Сначала обучите модель 1 на стуках и неподходящих звуках.")
            else:
                model = self.read_json("gate.json")
                gate_scores = self.scores(audio["gate_features"], model)
                accepted = gate_scores["knock"] >= model["threshold"]
                gate.update(status="accepted" if accepted else "rejected", accepted=accepted,
                            scores=gate_scores, threshold=model["threshold"], model_id=model["id"],
                            code="knock" if accepted else "not_knock",
                            reason="Подходящий стук обнаружен." if accepted else
                            "Не найден подходящий стук или балл детектора недостаточен. Запишите отдельный удар без речи и музыки.")
        prediction, scores = None, None
        if gate["status"] in {"accepted", "skipped"} and state["material_trained"]:
            scores = self.scores(audio["features"], self.read_json("model.json"))
            prediction = max(scores, key=scores.get)
        return gate, prediction, scores

    def analyze(self, raw, filename, mode="cascade", multiple=False):
        self.validate_mode(mode)
        if not isinstance(multiple, bool):
            raise ExperimentError("Параметр multiple должен быть логическим значением.")
        if multiple:
            impacts = []
            for impact in split_impacts(raw, filename):
                result = self.analyze(impact.pop("audio"), "impact.wav", mode)
                result["impact"] = impact
                impacts.append(result)
            return {"mode": mode, "multiple": True, "impacts": impacts}
        audio = process(raw, filename, allow_unsuitable=True)
        state = self.state()
        gate, prediction, scores = self.decide(audio, state, mode)
        rows, _ = self.scan()
        known = sorted(set(r["split"] for r in rows if r["audio"]["raw_hash"] == audio["raw_hash"] or
                           r["audio"]["canonical_hash"] == audio["canonical_hash"]))
        if known == ["demo"]:
            audio["warnings"].append("Эта запись сохранена вне оценки и не участвовала в обучении или проверке. Для независимой оценки используйте новый дубль.")
        elif known:
            audio["warnings"].append("Эта запись уже есть в данных: " + ", ".join({"demo": "вне оценки"}.get(split, split) for split in known) + ". Для независимой проверки используйте новый дубль.")
        return {"prediction": prediction, "scores": scores, "mode": mode, "gate": gate, "known_splits": known,
                "info": audio["info"], "warnings": audio["warnings"], "waveform": audio["waveform"],
                "spectrogram": audio["spectrogram"], "playback": base64.b64encode(playback(audio["samples"])).decode(),
                "model_id": state["model"]["id"] if prediction else None}

    def evaluate(self, mode="cascade"):
        self.validate_mode(mode)
        state = self.state()
        if not state["can_evaluate_material" if mode == "material" else "can_evaluate"]:
            raise ExperimentError("Нужны актуальные модели и минимум по 2 записи дерева и металла в test; для каскада также 2 неподходящих звука. Сессии должны отличаться от train.")
        rows, _ = self.scan()
        model = self.read_json("model.json")
        report_labels = DATA_CLASSES if mode == "cascade" else CLASSES
        test = [r for r in rows if r["split"] == "test" and r["label"] in report_labels]
        truth, predicted, details = [], [], []
        gate_truth, gate_predicted = [], []
        accepted_material_truth, accepted_material_predicted = [], []
        for row in test:
            gate, prediction, _ = self.decide(row["audio"], state, mode)
            result = prediction or "other"
            truth.append(row["label"])
            predicted.append(result)
            details.append({"path": row["path"], "actual": row["label"], "predicted": result,
                            "gate_status": gate["status"], "correct": row["label"] == result})
            if mode == "cascade":
                gate_truth.append("other" if row["label"] == "other" else "knock")
                gate_predicted.append("knock" if gate["accepted"] else "other")
                if row["label"] in CLASSES and prediction:
                    accepted_material_truth.append(row["label"])
                    accepted_material_predicted.append(prediction)
        n, acc = len(test), float(accuracy_score(truth, predicted))
        z = 1.96
        center = (acc + z*z/(2*n)) / (1 + z*z/n)
        radius = z * np.sqrt(acc*(1-acc)/n + z*z/(4*n*n)) / (1 + z*z/n)
        report = {"created": stamp(), "mode": mode, "model_id": model["id"],
                  "gate_id": state["gate_model"]["id"] if mode == "cascade" else None,
                  "test_signature": self.signature(rows, "test", report_labels),
                  "n_test": n, "accuracy": acc, "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
                  "macro_f1": float(f1_score(truth, predicted, labels=report_labels, average="macro", zero_division=0)),
                  "majority_baseline": max(Counter(truth).values()) / n,
                  "wilson95": [float(center-radius), float(center+radius)], "labels": report_labels,
                  "confusion": confusion_matrix(truth, predicted, labels=report_labels).tolist(),
                  "sessions": sorted(set(r["session"] for r in test)), "examples": details,
                  "warnings": state["warnings"], "file_split_sessions": self.file_split_sessions()}
        if mode == "cascade":
            negatives = sum(value == "other" for value in gate_truth)
            positives = len(gate_truth) - negatives
            false_accepts = sum(a == "other" and p == "knock" for a, p in zip(gate_truth, gate_predicted))
            false_rejects = sum(a == "knock" and p == "other" for a, p in zip(gate_truth, gate_predicted))
            report["gate"] = {"accuracy": float(accuracy_score(gate_truth, gate_predicted)),
                              "labels": ["knock", "other"],
                              "confusion": confusion_matrix(gate_truth, gate_predicted, labels=["knock", "other"]).tolist(),
                              "false_accepts": false_accepts, "n_other": negatives,
                              "false_rejects": false_rejects, "n_knocks": positives,
                              "false_accept_rate": false_accepts / negatives, "knock_recall": 1 - false_rejects / positives}
            report["material_accepted"] = {"n_test": len(accepted_material_truth),
                "coverage": len(accepted_material_truth) / positives,
                "accuracy": float(accuracy_score(accepted_material_truth, accepted_material_predicted)) if accepted_material_truth else None}
        self.write_json("report.json", report)
        return report
