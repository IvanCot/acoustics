"""Compare denoising with/without train augmentation on val, never score test."""
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acoustic.audio import PIPELINE_VERSION, process
from acoustic.experiment import Experiment, stamp


def validate(experiment, *, noisy=False):
    state = experiment.state()
    rows, _ = experiment.scan()
    examples = []
    for row in rows:
        if row['split'] != 'val':
            continue
        audio = row['audio']
        if noisy:
            raw = (experiment.data / row['path']).read_bytes()
            samples, rate = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
            rng = np.random.default_rng(int(hashlib.sha256(raw).hexdigest()[:16], 16))
            noise = rng.standard_normal(samples.shape)
            rms = np.sqrt(np.mean(samples ** 2))
            samples = samples + noise / max(np.sqrt(np.mean(noise ** 2)), 1e-12) * rms / 10
            buffer = io.BytesIO()
            sf.write(buffer, samples, rate, format='WAV', subtype='FLOAT')
            audio = process(buffer.getvalue(), 'noisy.wav', allow_unsuitable=True)
        gate, prediction, _ = experiment.decide(audio, state, 'cascade')
        material_scores = experiment.scores(audio['features'], experiment.read_json('model.json'))
        direct = max(material_scores, key=material_scores.get)
        examples.append({'path': row['path'], 'actual': row['label'],
                         'predicted': prediction or 'other', 'direct': direct,
                         'accepted': gate['accepted']})
    return {'n': len(examples),
            'correct': sum(r['actual'] == r['predicted'] for r in examples),
            'gate_correct': sum((r['actual'] != 'other') == r['accepted'] for r in examples),
            'material_correct': sum(r['actual'] == r['direct'] for r in examples if r['actual'] != 'other'),
            'n_material': sum(r['actual'] != 'other' for r in examples), 'examples': examples}


def main():
    output = ROOT / 'artifacts' / 'experiments' / 'denoise-augmentation'
    output.mkdir(parents=True, exist_ok=True)
    results = {'created': stamp(), 'pipeline': PIPELINE_VERSION, 'split': 'val',
               'stress': 'Deterministic added white noise at 20 dB SNR; synthetic robustness check only.',
               'limitations': 'Shared capture sessions and source differences can inflate these metrics.'}
    for name, augment in [('denoise', False), ('denoise_augmented', True)]:
        experiment = Experiment(ROOT / 'data', output / name)
        experiment.train(augment=augment)
        results[name] = {'clean': validate(experiment), 'noisy': validate(experiment, noisy=True)}
        print(name, {k: {key: val for key, val in v.items() if key != 'examples'}
                     for k, v in results[name].items()}, flush=True)
    # Keep augmented models when their clean validation score is not worse.
    selected = ('denoise_augmented' if results['denoise_augmented']['clean']['correct'] >=
                results['denoise']['clean']['correct'] else 'denoise')
    results['selected'] = selected
    before = output / 'before-validation.json'
    if before.exists():
        results['before'] = json.loads(before.read_text('utf-8'))
    for name in ['model.json', 'gate.json']:
        shutil.copy2(output / selected / name, ROOT / 'artifacts' / name)
    (ROOT / 'artifacts' / 'validation.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('selected', selected, flush=True)


if __name__ == '__main__':
    main()
