#!/usr/bin/env python3
"""
eval_feature_fusion.py — Evaluates B1, B2, B3, B4 configurations using FEATURE-LEVEL FUSION
across 3 supported BLIP retriever backbone models:
  1. 'chatir' : ChatIR Fine-tuned on Visual Dialog (chatir_weights.ckpt)
  2. 'cocoft' : COCO Retrieval Fine-tuned (model_base_retrieval_coco.pth)
  3. 'zs'     : Pure Zero-Shot Pretrain (model_base.pth - 129M images)

Fusion Mechanism:
  v_fused = w_a * v_text + w_b * v_gen + w_g * v_sketch
  v_query = F.normalize(v_fused, dim=-1)
  s = v_query @ C.T

4 Retrieval Configurations:
  B1  Text-only               : v_query = v_text
  B2  Direct Sketch           : v_query = Normalize(w_a * v_text + w_g * v_sketch)
  B3  DAR (Text + Gen)        : v_query = Normalize(w_a * v_text + w_b * v_gen)
  B4  SG-DAR (Text+Gen+Sketch): v_query = Normalize(w_a * v_text + w_b * v_gen + w_g * v_sketch)
"""

import argparse
import hashlib
import json
import os
import sys

import torch
import torch.nn.functional as F
import tqdm

from baselines import BLIP_BASELINE, BLIP_CKPTS

MODEL_NAMES = {
    'chatir': 'ChatIR Fine-tuned BLIP (chatir_weights.ckpt)',
    'cocoft': 'COCO-FT BLIP (model_base_retrieval_coco.pth)',
    'zs':     'Zero-Shot BLIP 129M (model_base.pth)',
}

CONFIGS = {
    'B1': dict(name='Text-only (B1)',                   gen=False, sketch=False),
    'B2': dict(name='Direct Sketch [Text+Sketch] (B2)',  gen=False, sketch=True),
    'B3': dict(name='DAR [Text+Gen] (B3)',               gen=True,  sketch=False),
    'B4': dict(name='SG-DAR [Text+Gen+Sketch] (B4)',     gen=True,  sketch=True),
}


def _find_file(path, extra_dirs=('dataset', 'ChatIR_Protocol', 'dialogues', '.')):
    """Find file on disk across fallback directories."""
    if not path:
        return path
    if os.path.exists(path):
        return path
    base = os.path.basename(path)
    for d in extra_dirs:
        cand = os.path.join(d, base)
        if os.path.exists(cand):
            return cand
    if os.path.exists(base):
        return base
    return path


def _tag(path, n=8):
    """Generate content fingerprint hash to ensure cache is loaded correctly."""
    try:
        real_p = _find_file(path)
        stem = os.path.splitext(os.path.basename(real_p))[0]
        h = hashlib.md5(open(real_p, 'rb').read()).hexdigest()[:n]
        return f"{stem}_{h}"
    except Exception:
        return os.path.splitext(os.path.basename(path))[0]


# ============================================================ Datasets

class Corpus(torch.utils.data.Dataset):
    """50k candidate images dataset."""

    def __init__(self, corpus_path, preprocessor):
        corpus_path = _find_file(corpus_path)
        with open(corpus_path) as f:
            self.corpus = json.load(f)
        self.preprocessor = preprocessor
        self.path2id = {self.corpus[i]: i for i in range(len(self.corpus))}

    def __len__(self):
        return len(self.corpus)

    def path_to_index(self, path):
        return self.path2id[path]

    def __getitem__(self, i):
        return {'idx': i, 'image': self.preprocessor(self.corpus[i]), 'valid': 1}


class ImageBank(torch.utils.data.Dataset):
    """
    Manages image index by (dialog, round) -> idx = d * rounds + r.
    mode='gen'    : reads {root}/{d}_{r}.jpg
    mode='sketch' : reads 'sketch' field in queries JSON.
    """

    def __init__(self, queries_path, preprocessor, root, mode, rounds=11):
        queries_path = _find_file(queries_path)
        with open(queries_path) as f:
            self.queries = json.load(f)
        self.preprocessor = preprocessor
        self.root = root or ''
        self.mode = mode
        self.rounds = rounds
        self.paths = self._resolve()
        self.n_missing = sum(1 for p in self.paths if p is None)

    def _find(self, raw):
        if not raw:
            return None
        cands = [raw]
        if self.root:
            cands.append(os.path.join(self.root, raw))
        for c in list(cands):
            stem = os.path.splitext(c)[0]
            cands += [stem + e for e in ('.png', '.jpg', '.jpeg', '.webp')]
        for c in cands:
            if os.path.exists(c):
                return c
        return None

    def _resolve(self):
        paths = []
        for d, item in enumerate(self.queries):
            turns = item.get('dialog', [])
            for r in range(self.rounds):
                if self.mode == 'gen':
                    cand = os.path.join(self.root, f'{d}_{r}.jpg')
                    if not os.path.exists(cand):
                        found = None
                        for ext in ('.png', '.jpeg', '.webp'):
                            alt = os.path.join(self.root, f'{d}_{r}{ext}')
                            if os.path.exists(alt):
                                found = alt
                                break
                        paths.append(found)
                    else:
                        paths.append(cand)
                    continue

                raw = ''
                if r < len(turns) and isinstance(turns[r], dict):
                    raw = turns[r].get('sketch', '')
                paths.append(self._find(raw))
        return paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        if p is None:
            return {'idx': idx, 'image': torch.zeros(3, 224, 224), 'valid': 0}
        try:
            return {'idx': idx, 'image': self.preprocessor(p), 'valid': 1}
        except Exception:
            return {'idx': idx, 'image': torch.zeros(3, 224, 224), 'valid': 0}


class Queries(torch.utils.data.Dataset):
    """Text queries at each turn."""

    def __init__(self, cfg, queries_path):
        queries_path = _find_file(queries_path)
        with open(queries_path) as f:
            self.queries = json.load(f)
        self.cfg = cfg
        self.dialog_length = None

    @staticmethod
    def _text(turn):
        return turn['text'] if isinstance(turn, dict) else str(turn)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        assert self.dialog_length is not None, 'Must set self.dialog_length first.'
        turns = self.queries[i]['dialog']
        r = min(self.dialog_length, len(turns) - 1)
        text = self._text(turns[r])
        return {'text': text, 'target_path': self.queries[i]['img'], 'idx': i}


# ============================================================ Feature Encoding & Caching

def encode_bank(dataset, embedder, cfg, cache=''):
    """Encodes image dataset -> (feats, valid_mask) sorted by idx, with caching support."""
    if cache and os.path.exists(cache) and not cfg.get('overwrite_cache', False):
        try:
            print(f'  [cache] {cache}')
            feats, valid = torch.load(cache, map_location=cfg['device'])
            return feats.to(cfg['device']), valid.to(cfg['device'])
        except Exception as e:
            print(f'  ⚠ Cache loading error ({e}) -> Re-extracting features...')

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], pin_memory=True
    )

    feats, ids, valids = [], [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, leave=False, ncols=80):
            v = F.normalize(embedder.model(batch['image'].to(cfg['device'])), dim=-1)
            feats.append(v)
            ids.append(batch['idx'].to(cfg['device']))
            valids.append(batch['valid'].to(cfg['device']))

    feats = torch.cat(feats)
    ids = torch.cat(ids)
    valids = torch.cat(valids).bool()
    order = torch.argsort(ids)
    out = (feats[order], valids[order])

    if cache:
        try:
            os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
            torch.save((out[0].cpu(), out[1].cpu()), cache)
        except Exception:
            pass
    return out


# ============================================================ Metrics

def get_first_hitting_time(target_recall, hitting_recall=10, rounds=11):
    recalls = target_recall.view(rounds, -1).T
    hit = recalls < hitting_recall
    final = torch.inf * torch.ones(recalls.shape[0])
    hist = []
    for r in range(rounds):
        sel = hit[:, r]
        final[sel] = torch.min(final[sel], torch.ones(final[sel].shape) * r)
        hist.append(final.clone())
    return torch.stack(hist)


def cumulative_hits_per_round(target_recall, hitting_recall=10, rounds=11):
    ht = get_first_hitting_time(target_recall, hitting_recall, rounds)
    return (ht < torch.inf).sum(dim=-1) * 100 / ht[0].shape[0]


# ============================================================ Feature Fusion Evaluator

class FeatureFusionEvaluator:

    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = CONFIGS[cfg['config']]

        self.dialog_encoder, self.blip_img = BLIP_BASELINE(
            device=cfg['device'],
            ckpt=cfg.get('blip_ckpt', None),
            mode=cfg.get('model_type', 'chatir')
        )
        self.corpus_ds = Corpus(cfg['corpus_path'], self.blip_img.processor)

        self.corpus_blip = None
        self.gen_feats = self.gen_valid = None
        self.sk_feats = self.sk_valid = None

    def index(self):
        m_tag = self.cfg.get('model_tag', 'chatir')
        print(f'[index] corpus / BLIP ({m_tag})')
        self.corpus_blip = encode_bank(
            self.corpus_ds, self.blip_img, self.cfg,
            cache=os.path.join(self.cfg['cache_dir'], f'corpus_{m_tag}.pth')
        )[0]

        q_tag = _tag(self.cfg['queries_path'])

        if self.mode['gen']:
            tag = os.path.basename(self.cfg['gen_path'].rstrip('/'))
            print(f'[index] generated images / BLIP ({m_tag} · {tag} · {q_tag})')
            ds = ImageBank(
                self.cfg['queries_path'], self.blip_img.processor,
                self.cfg['gen_path'], 'gen', self.cfg['rounds']
            )
            if ds.n_missing:
                print(f'  ⚠ missing {ds.n_missing}/{len(ds)} generated images')
            self.gen_feats, self.gen_valid = encode_bank(
                ds, self.blip_img, self.cfg,
                cache=os.path.join(self.cfg['cache_dir'], f'gen_{m_tag}_{tag}_{q_tag}.pth')
            )

        if self.mode['sketch']:
            print(f'[index] sketch / BLIP ({m_tag} · {q_tag})')
            ds = ImageBank(
                self.cfg['queries_path'], self.blip_img.processor,
                self.cfg['sketch_root'], 'sketch', self.cfg['rounds']
            )
            self.sk_feats, self.sk_valid = encode_bank(
                ds, self.blip_img, self.cfg,
                cache=os.path.join(self.cfg['cache_dir'], f'sketch_{m_tag}_{q_tag}.pth')
            )
            n = int(self.sk_valid.sum())
            print(f'  {n}/{len(self.sk_valid)} turns with valid sketches')
            if n == 0:
                raise RuntimeError('No valid sketches found — check --sketch_root')

    def base_weights(self, r):
        if not self.mode['gen']:
            return 1.0, 0.0
        return (0.8, 0.2) if r < 2 else (0.5, 0.5)      # (text, gen)

    def _ranks(self, loader, r):
        loader.dataset.dialog_length = r
        out = []
        dev = self.cfg['device']
        a0, b0 = self.base_weights(r)
        g0 = self.cfg['gamma'] if self.mode['sketch'] else 0.0

        for batch in tqdm.tqdm(loader, leave=False, ncols=80):
            tgt = torch.tensor(
                [self.corpus_ds.path_to_index(p) for p in batch['target_path']],
                device=dev
            ).unsqueeze(1)
            qid = batch['idx'].to(dev)
            bank = qid * self.cfg['rounds'] + r

            # 1. Text feature extraction
            tvec = F.normalize(self.dialog_encoder(batch['text']), dim=-1)

            # 2. Sketch validity mask
            if self.mode['sketch']:
                m = self.sk_valid[bank].float().unsqueeze(1)
            else:
                m = torch.zeros(len(qid), 1, device=dev)

            # 3. Compute weights for each modality
            wa = a0 * (1.0 - g0 * m)
            wb = b0 * (1.0 - g0 * m)
            wg = g0 * m

            # 4. FEATURE-LEVEL FUSION
            fused_vec = wa * tvec

            if self.mode['gen']:
                gv = self.gen_feats[bank]
                gm = self.gen_valid[bank].float().unsqueeze(1)
                fused_vec = fused_vec + (wb * gm) * gv

            if self.mode['sketch']:
                sv = self.sk_feats[bank]
                fused_vec = fused_vec + wg * sv

            # 5. L2 normalize fused query vector
            query_vec = F.normalize(fused_vec, dim=-1)

            # 6. Compute Cosine Similarity against all 50k Corpus images
            scores = query_vec @ self.corpus_blip.T

            # 7. Ranking
            ranks = torch.argsort(scores, descending=True, dim=1).long()
            out.append(((ranks - tgt) == 0).nonzero()[:, 1])

        return torch.cat(out)

    def run(self, hits_at=10):
        ds = Queries(self.cfg, self.cfg['queries_path'])
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.cfg['batch_size'], shuffle=False,
            num_workers=self.cfg['num_workers'], pin_memory=True
        )

        all_ranks = [self._ranks(loader, r) for r in range(self.cfg['rounds'])]
        hits = cumulative_hits_per_round(
            torch.cat(all_ranks).cpu(), hits_at, self.cfg['rounds']
        ).tolist()

        m_name = MODEL_NAMES.get(self.cfg.get('model_type', 'chatir'), self.cfg.get('model_tag', 'chatir').upper())
        print(f"\n----- [{m_name}] · {self.cfg['config']} ({self.mode['name']}) · Hits@{hits_at} -----")
        for r in range(self.cfg['rounds']):
            print(f'  Turn {r:>2}: {hits[r]:6.2f}%')
        return hits


# ============================================================ Main

def run_eval_for_model(model_type, args, base_cfg, todo_configs):
    ckpt = args.blip_ckpt if args.blip_ckpt else BLIP_CKPTS.get(model_type, 'chatir_weights.ckpt')
    mode_tag = model_type if not args.blip_ckpt else 'custom'
    cache_dir = args.cache_dir if args.cache_dir else f'temp_{mode_tag}'
    model_desc = MODEL_NAMES.get(model_type, model_type.upper())

    print(f"\n{'#'*68}")
    print(f"🚀 RUNNING FEATURE FUSION EVALUATION — MODEL: {model_desc}")
    print(f"   Model Identifier: {model_type}")
    print(f"   BLIP Checkpoint : {ckpt}")
    print(f"   Cache Directory : {cache_dir}/")
    print(f"{'#'*68}")

    model_cfg = dict(base_cfg,
                     model_type=model_type,
                     blip_ckpt=ckpt,
                     model_tag=mode_tag,
                     cache_dir=cache_dir)

    results = {}
    with torch.no_grad():
        for c in todo_configs:
            cfg = dict(model_cfg, config=c)
            cfg['gen_path'] = args.gen_path_dar if c == 'B3' else args.gen_path
            print(f"\n{'='*65}\n[{model_type.upper()}] {c} — {CONFIGS[c]['name']}\n{'='*65}")
            ev = FeatureFusionEvaluator(cfg)
            ev.index()
            results[c] = ev.run(hits_at=args.hits_at)

    if len(results) > 1:
        print(f"\n{'='*65}\nSUMMARY [{model_desc}] — Hits@{args.hits_at} (%)\n{'='*65}")
        print('Turn   ' + ''.join(f'{c:>10}' for c in todo_configs))
        for r in range(args.rounds):
            print(f'{r:>4}   ' + ''.join(f'{results[c][r]:10.2f}' for c in todo_configs))
        if 'B3' in results and 'B4' in results:
            d = results['B4'][-1] - results['B3'][-1]
            print(f'\nB4 (SG-DAR) − B3 (DAR) at final turn (R10): {d:+.2f} points (sketch contribution via Feature Fusion)')
    return results


def main():
    p = argparse.ArgumentParser(
        description='Evaluate retrieval using Feature-Level Fusion across 3 models: chatir, cocoft, zs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model_type', choices=['chatir', 'cocoft', 'zs', 'all'], default='chatir',
                   help="Model to evaluate: 'chatir' (VisDial fine-tuned), 'cocoft' (COCO fine-tuned), 'zs' (Pure 129M Zero-Shot), or 'all'.")
    p.add_argument('--config', choices=list(CONFIGS) + ['all'], default='all',
                   help="Evaluation configuration: B1, B2, B3, B4, or all.")
    p.add_argument('--blip_ckpt', default='',
                   help='Custom checkpoint URL or file path (overrides --model_type preset).')
    p.add_argument('--queries_path', default='VisDial_v1_0_queries_val_sketch.json',
                   help='Path to queries JSON file.')
    p.add_argument('--corpus_path', default='ChatIR_Protocol/Search_Space_val_50k.json',
                   help='Path to 50k corpus images JSON file.')
    p.add_argument('--gen_path', default='generated_images_text_sketch',
                   help='Generated images directory for B4 (text + sketch).')
    p.add_argument('--gen_path_dar', default='generated_images_text',
                   help='Generated images directory for B3 (pure text).')
    p.add_argument('--sketch_root', default='',
                   help='Root directory for sketch images (if empty, resolves relative to JSON).')
    p.add_argument('--cache_dir', default='',
                   help='Directory to store cached .pth feature vectors (defaults to temp_<model_type>).')
    p.add_argument('--gamma', type=float, default=0.2,
                   help='Weight for the sketch branch.')
    p.add_argument('--rounds', type=int, default=11,
                   help='Number of dialogue turns to evaluate.')
    p.add_argument('--batch_size', type=int, default=128,
                   help='Batch size for feature extraction.')
    p.add_argument('--num_workers', type=int, default=4,
                   help='Number of PyTorch DataLoader worker threads.')
    p.add_argument('--hits_at', type=int, default=10,
                   help='Threshold for Hits@k metric.')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                   help="Computing device: 'auto' (prefer CUDA), 'cuda', or 'cpu'.")
    p.add_argument('--overwrite_cache', action='store_true',
                   help='Ignore existing cache and force re-extracting feature vectors.')
    a = p.parse_args()

    dev = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device

    base_cfg = dict(
        queries_path=a.queries_path, corpus_path=a.corpus_path,
        sketch_root=a.sketch_root,
        gamma=a.gamma, rounds=a.rounds,
        batch_size=a.batch_size, num_workers=a.num_workers,
        device=dev,
        overwrite_cache=a.overwrite_cache,
    )

    todo_configs = list(CONFIGS) if a.config == 'all' else [a.config]
    models_to_run = ['chatir', 'cocoft', 'zs'] if a.model_type == 'all' else [a.model_type]

    for m in models_to_run:
        run_eval_for_model(m, a, base_cfg, todo_configs)


if __name__ == '__main__':
    main()
