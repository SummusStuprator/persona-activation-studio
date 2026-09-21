"""Paper-inspired measurement with train-only PCA and grouped held-out tests."""
from __future__ import annotations
import os
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from backend import ROOT, ABI, Engine, save_run
PAPER_COMMIT = '8d1649c03a63a39c9aa092532c376800cc4a3863'
PAIN = {'A1', 'A2', 'A3', 'A4', 'A5'}
CONTROL_NAMES = {'B': 'fear', 'C1': 'negative_emotion', 'C2': 'negative_world', 'E': 'bodily_sensation'}

def paper_data() -> dict:
    p = ROOT / 'paper/datasets/3.1_pain_and_control_datasets.json'
    return json.loads(p.read_text(encoding='utf-8'))['datasets']

def canonical_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()

def rows_for(name: str) -> list[dict]:
    return [{'text': r['prompt'], 'category': r['category'], 'group': str(r['set'])}
            for r in paper_data()[name]['sentences']]

def collect(engine, rows, progress=None, pooling='last'):
    from workshop_v2.universal_training import collect_resumable
    return collect_resumable(engine, rows, progress, pooling)


def direction(x: np.ndarray, y: np.ndarray, denoise: bool = True) -> np.ndarray:
    positive, negative = x[y].astype(np.float64), x[~y].astype(np.float64)
    v = positive.mean(0) - negative.mean(0)
    if denoise and len(negative) > 1:
        centered = negative - negative.mean(0)
        eigenvalues, eigenvectors = np.linalg.eigh(centered @ centered.T)
        order = np.argsort(eigenvalues)[::-1]
        values, vectors = eigenvalues[order], eigenvectors[:, order]
        keep = values > max(float(values[0]) * 1e-10, 1e-12)
        values, vectors = values[keep], vectors[:, keep]
        if len(values):
            k = int(np.searchsorted(np.cumsum(values), 0.5 * values.sum())) + 1
            components = (vectors[:, :k].T @ centered) / np.sqrt(values[:k, None])
            v -= components.T @ (components @ v)
    norm = np.linalg.norm(v)
    if norm < 1e-10: raise ValueError('Degenerate contrast: use more diverse examples.')
    return (v / norm).astype(np.float32)

def fit_axis(engine: Engine, rows: list[dict], x: np.ndarray, labels: np.ndarray,
             name: str, source: str, progress=None, pooling: str = 'last') -> dict:
    labels = np.asarray(labels, bool)
    groups = np.array([r['group'] for r in rows])
    unique = np.unique(groups)
    if len(unique) < 10: raise ValueError('At least 10 independent contrast groups are required.')
    rng = np.random.default_rng(20260919)
    shuffled = rng.permutation(unique)
    heldout_groups = shuffled[:max(2, int(np.ceil(len(unique) * 0.2)))]
    train = np.flatnonzero(~np.isin(groups, heldout_groups))
    test = np.flatnonzero(np.isin(groups, heldout_groups))
    if len(np.unique(labels[train])) != 2 or len(np.unique(labels[test])) != 2:
        raise ValueError('Both classes must occur in training and held-out groups.')
    folds = list(GroupKFold(n_splits=5).split(train, labels[train], groups[train]))
    directions, centers, scales, norms, cv_scores = [], [], [], [], []
    with threadpool_limits(limits=max(1,int(os.environ.get("WORKSHOP_CPU_THREADS","2")))):
        for layer in range(engine.layers):
            scores = []
            for fit, validate in folds:
                v = direction(x[train[fit], layer], labels[train[fit]])
                scores.append(roc_auc_score(labels[train[validate]], x[train[validate], layer] @ v))
            v = direction(x[train, layer], labels[train])
            projection = x[train, layer] @ v
            directions.append(v)
            centers.append(projection.mean())
            scales.append(max(float(projection.std(ddof=1)), 1e-8))
            norms.append(np.linalg.norm(x[train, layer], axis=1).mean())
            cv_scores.append(float(np.mean(scores)))
            if progress: progress(layer + 1, engine.layers, 'Train-only grouped cross-validation: ' + name)
    chosen = int(np.argmax(cv_scores))
    heldout_projection = x[test, chosen] @ directions[chosen]
    auc = float(roc_auc_score(labels[test], heldout_projection))
    raw_direction = direction(x[train, chosen], labels[train], denoise=False)
    raw_auc = float(roc_auc_score(labels[test], x[test, chosen] @ raw_direction))
    shuffled_labels = rng.permutation(labels[train])
    shuffled_direction = direction(x[train, chosen], shuffled_labels)
    shuffle_auc = float(roc_auc_score(labels[test], x[test, chosen] @ shuffled_direction))
    meta = {'name': name, 'source': source, 'model': engine.model, 'abi': getattr(engine, 'abi', ABI),
            'paper_commit': PAPER_COMMIT, 'dataset_hash': canonical_hash(rows), 'pooling': pooling,
            'layer': chosen, 'train_n': len(train), 'test_n': len(test), 'cv_auc': cv_scores,
            'heldout_auc': auc, 'raw_difference_heldout_auc': raw_auc, 'shuffled_label_control_auc': shuffle_auc,
            'heldout_groups': heldout_groups.tolist(), 'created_epoch': time.time(),
            'method': 'Train-only PCA denoising, five-fold grouped layer selection, independent grouped test.',
            'interpretation': 'A text-representation measurement, not a probability of emotion.',
            'deviations': 'Local backend '+str(engine.model.get('backend','native_gguf'))+'; train-only final fit, norm-fraction doses; no new weight fine-tuning.'}
    result = {'directions': np.stack(directions), 'center': np.array(centers, np.float32),
              'scale': np.array(scales, np.float32), 'norm': np.array(norms, np.float32),
              'layer': chosen, 'meta': meta}
    folder = ROOT / 'vectors' / engine.model['digest']
    folder.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r'[^a-z0-9_]+', '_', name.lower()).strip('_')[:48]
    if not slug: raise ValueError('Give this direction a name.')
    arrays = {k: result[k] for k in ('directions', 'center', 'scale', 'norm')}
    np.savez_compressed(folder / (slug + '.npz'), **arrays)
    meta['arrays_sha256'] = hashlib.sha256((folder / (slug + '.npz')).read_bytes()).hexdigest()
    meta['training_backend'] = engine.model.get('backend','native_gguf')
    (folder / (slug + '.json')).write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return result

from bank_loader import load_bank

def train_paper(engine: Engine, versions=('S2', 'S1'), progress=None, pooling='last') -> dict:
    for version in versions:
        rows = rows_for(version + '_1P')
        x = collect(engine, rows, progress, pooling)
        labels = np.array([r['category'] in PAIN for r in rows])
        fit_axis(engine, rows, x, labels, 'pain_' + version.lower(),
                 'Official paper dataset: ' + version + '_1P', progress, pooling)
        if version == 'S2':
            for category, name in CONTROL_NAMES.items():
                indices = [i for i, r in enumerate(rows) if r['category'] in (category, 'D')]
                subrows = [rows[i] for i in indices]
                sublabels = np.array([r['category'] == category for r in subrows])
                fit_axis(engine, subrows, x[indices], sublabels, name,
                         'Local S2-only control contrast against neutral D; not the full pooled paper control fit.',
                         progress, pooling)
    return load_bank(engine)

def measure(engine: Engine, text: str, bank: dict) -> dict:
    h = engine.extract(text)
    profiles, scores = {}, {}
    for name, axis in bank.items():
        z = (np.einsum('ld,ld->l', h, axis['directions']) - axis['center']) / axis['scale']
        profiles[name] = z.tolist()
        scores[name] = float(z[axis['layer']])
    result = {'model': engine.model, 'text': text, 'scores': scores, 'layer_profiles': profiles,
              'residual_norms': np.linalg.norm(h, axis=1).tolist(), 'abi': getattr(engine, 'abi', ABI)}
    file = save_run(result, 'probe')
    np.savez_compressed(file.with_suffix('.npz'), residual=h)
    result['saved_to'] = str(file)
    return result

def scenario_messages(text: str) -> list[dict]:
    parts = re.split(r'\[(User|Assistant|System)\]:', text)
    messages = []
    for index in range(1, len(parts) - 1, 2):
        content = parts[index + 1].strip()
        if content:
            messages.append({'role': parts[index].lower(), 'content': content})
    if not messages:
        raise ValueError('Unrecognized paper scenario format.')
    return messages

def screen_scenarios(engine: Engine, bank: dict, per_category: int = 2, progress=None) -> list[dict]:
    path = ROOT / 'paper/datasets/4.1_self_other_420_scenarios.json'
    scenarios = json.loads(path.read_text(encoding='utf-8'))
    counts, selected, results = {}, [], []
    for row in scenarios:
        category = row['category']
        if counts.get(category, 0) < per_category:
            selected.append(row)
            counts[category] = counts.get(category, 0) + 1
    for i, row in enumerate(selected):
        prompt = engine.chat(scenario_messages(row['text']))
        probe = measure(engine, prompt, bank)
        results.append({'id': row['id'], 'category': row['category'], 'stratum': row['stratum'], **probe['scores']})
        if progress:
            progress(i + 1, len(selected), 'Unsteered paper scenario measurements')
    save_run({'model': engine.model, 'per_category': per_category, 'rows': results}, 'scenario-screen')
    return results

def train_custom(engine: Engine, positive: list[str], negative: list[str], name: str, progress=None) -> dict:
    if len(positive) != len(negative) or len(positive) < 20:
        raise ValueError('Supply at least 20 matched positive/negative examples; counts must match.')
    if any(not t.strip() for t in positive + negative):
        raise ValueError('Examples cannot be empty.')
    if len(set(positive)) != len(positive) or len(set(negative)) != len(negative):
        raise ValueError('Use distinct examples, not duplicates.')
    rows = []
    for i, (pos, neg) in enumerate(zip(positive, negative)):
        rows.extend([{'text': pos, 'category': 'positive', 'group': str(i)},
                     {'text': neg, 'category': 'negative', 'group': str(i)}])
    x = collect(engine, rows, progress)
    labels = np.array([r['category'] == 'positive' for r in rows])
    return fit_axis(engine, rows, x, labels, name,
                    'User-defined local contrast. Not an SAE feature or an Anthropic feature.', progress)

def control_screen(engine: Engine, bank: dict, sample_size: int = 40, progress=None) -> list[dict]:
    if 'pain_s2' not in bank:
        raise ValueError('Build the paper S2 direction first.')
    datasets = paper_data()
    sadness_path = ROOT / 'paper/datasets/3.1_sadness_dataset.json'
    datasets.update(json.loads(sadness_path.read_text(encoding='utf-8'))['datasets'])
    names = ['S2_3P', 'Numb_1P', 'Numb_3P', 'Random_1P', 'Arousal_1P', 'SD_sadness_1P']
    results = []
    for name in names:
        source = datasets[name]['sentences']
        stride = max(1, len(source) // sample_size)
        source = source[::stride][:sample_size]
        rows = [{'text': r['prompt'], 'category': r['category'], 'group': str(r['set'])} for r in source]
        x = collect(engine, rows, progress)
        result = {'dataset': name, 'n': len(rows)}
        for key, axis in bank.items():
            layer = axis['layer']
            z = (x[:, layer] @ axis['directions'][layer] - axis['center'][layer]) / axis['scale'][layer]
            result[key + ' mean z'] = float(z.mean())
            if key == 'pain_s2':
                result['pain_s2 standard deviation'] = float(z.std(ddof=1))
        results.append(result)
    save_run({'model': engine.model, 'sample_size': sample_size, 'rows': results}, 'control-screen')
    return results
