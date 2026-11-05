"""Reproducible robustness and negative-control audits for the *toy* lab.

No generative model, network, training, or live checkpoint mutation is involved.
All patch selection uses the validation split; test and audit are measured later.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from statistics import mean

from .benchmarks import evaluate
from .checkpoints import fingerprint, write_json
from .data import samples
from .models import corrupted_model, model_from_state
from .pipeline import verify_exact
from .safety import PatchRejected, apply_checked, validation_gate
from .search import propose


def run_sweep(out: str | Path, base_seed: int = 42, seeds: int = 12, count: int = 1024) -> dict:
    """Independent seeded split draws; select only on each seed's validation split."""
    if not 1 <= seeds <= 1000:
        raise ValueError('seeds must be 1..1000')
    if count < 20:
        raise ValueError('count must be >=20')
    records = []
    for seed in range(base_seed, base_seed + seeds):
        base = corrupted_model().state_dict()
        base_sha = fingerprint(base)
        patch, evidence = propose(base, samples('validation', seed=seed, count=count))
        edited = apply_checked(base, patch)
        before_validation = evaluate(model_from_state(base), samples('validation', seed=seed, count=count))
        after_validation = evaluate(model_from_state(edited), samples('validation', seed=seed, count=count))
        validation_gate(before_validation, after_validation)
        verified = verify_exact(base, edited, patch)
        if fingerprint(base) != base_sha:
            raise RuntimeError('Base state was modified during audit')
        result = {'seed': seed, 'changed_elements': verified['changed_elements'],
                  'selection_split': evidence['selection_split'],
                  'selected_value': patch['edits'][0]['to'],
                  'candidates_evaluated': evidence['candidates_evaluated']}
        for split in ('validation', 'test', 'audit'):
            draw = samples(split, seed=seed, count=count)
            result[split] = {'before': evaluate(model_from_state(base), draw).accuracy,
                             'after': evaluate(model_from_state(edited), draw).accuracy}
        records.append(result)
    stats = {}
    for split in ('validation', 'test', 'audit'):
        stats[split] = {}
        for side in ('before', 'after'):
            values = [r[split][side] for r in records]
            stats[split][side] = {'mean': mean(values), 'min': min(values), 'max': max(values)}
    report = {'experiment': 'toy-linear-repair', 'seed_start': base_seed,
              'seeds': seeds, 'samples_per_split': count,
              'rule': 'Each patch selected on validation only; test and audit scored afterward',
              'summary': stats, 'runs': records}
    out = Path(out)
    write_json(out / 'sweep.json', report)
    lines = ['# Multi-seed synthetic robustness audit', '',
             '**Scope:** one known corrupted coordinate in a synthetic classifier, not Qwen.',
             f'**Seeds:** {seeds} from {base_seed}; {count} examples per split per seed.',
             '', '| Split | Before mean | After mean | After range |', '|---|---:|---:|---:|']
    for split in ('validation', 'test', 'audit'):
        b, a = stats[split]['before'], stats[split]['after']
        lines.append(f"| {split} | {b['mean']:.2%} | {a['mean']:.2%} | {a['min']:.2%}–{a['max']:.2%} |")
    lines += ['', '- No test/audit observations were consulted in patch selection.',
              '- The rule and the corrupt coordinate are known by construction.',
              '- Perfect outcomes on this synthetic rule do not imply real-world generalization.', '']
    (out / 'sweep.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def run_negative_controls(out: str | Path) -> dict:
    """Make unsafe edits; demonstrate that policy/validation detects every case."""
    base = corrupted_model().state_dict()
    original_sha = fingerprint(base)
    patch, _ = propose(base, samples('validation'))
    cases: list[tuple[str, dict]] = []
    p = deepcopy(patch); p['base_sha256'] = '0'*64
    cases.append(('stale_base', p))
    p = deepcopy(patch); p['edits'][0]['tensor'] = 'linear.bias'; p['edits'][0]['index'] = [0]
    cases.append(('forbidden_tensor', p))
    p = deepcopy(patch); p['edits'][0]['index'] = [90, 0]
    cases.append(('out_of_bounds', p))
    p = deepcopy(patch); p['edits'][0]['to'] = float('nan')
    cases.append(('non_finite', p))
    p = deepcopy(patch); p['edits'][0]['from'] = 999.0
    cases.append(('wrong_original_value', p))
    p = deepcopy(patch); p['edits'][0]['to'] = -2.0
    cases.append(('no_op', p))
    p = deepcopy(patch); p['edits'][0]['to'] = 7.0
    cases.append(('oversized_delta', p))
    p = deepcopy(patch); p['edits'][0]['index'] = [True, 0]
    cases.append(('boolean_index', p))
    p = deepcopy(patch); p['schema_version'] = True
    cases.append(('boolean_schema_version', p))
    result = []
    for name, candidate in cases:
        try:
            apply_checked(base, candidate)
            caught, reason = False, 'Unexpectedly accepted'
        except PatchRejected as exc:
            caught, reason = True, str(exc)
        result.append({'case': name, 'rejected': caught, 'reason': reason})
    # This edit conforms to numeric bounds but is rejected by the accuracy gate.
    p = deepcopy(patch); p['edits'][0]['to'] = -3.0
    try:
        edited = apply_checked(base, p)
        val = samples('validation')
        validation_gate(evaluate(model_from_state(base), val),
                        evaluate(model_from_state(edited), val))
        caught, reason = False, 'Unexpectedly accepted'
    except PatchRejected as exc:
        caught, reason = True, str(exc)
    result.append({'case': 'valid_format_regressive_performance', 'rejected': caught,
                   'reason': reason})
    if fingerprint(base) != original_sha:
        raise RuntimeError('An adversarial proposal modified the base checkpoint')
    report = {'experiment': 'toy-patch-negative-controls', 'base_unchanged': True,
              'controls': len(result), 'rejected': sum(r['rejected'] for r in result),
              'passed': all(r['rejected'] for r in result), 'cases': result,
              'warning': 'These are local policy checks, NOT an enterprise security certification.'}
    out = Path(out)
    write_json(out / 'negative_controls.json', report)
    lines = ['# Patch rejection audit', '',
             f"**Outcome:** {report['rejected']}/{report['controls']} unsafe or regressive edits rejected.",
             '', '| Attempt | Result |', '|---|---|']
    for row in result:
        lines.append(f"| {row['case']} | {'REJECTED' if row['rejected'] else 'ACCEPTED (FAIL)'} |")
    lines.extend(['', 'The starting checkpoint hash remained unchanged.',
                  '**Limitation:** This is a targeted regression suite, not a full threat-model assessment.', ''])
    (out / 'negative_controls.md').write_text('\n'.join(lines), encoding='utf-8')
    return report
