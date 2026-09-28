#!/usr/bin/env python3
"""Portable launcher for the frozen two-scenario ORPG configurations."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
from string import Template
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]

def build_command(scenario, values, extra=()):
    config = yaml.safe_load((ROOT / 'configs' / f'{scenario}.yaml').read_text())
    overrides = [Template(x).substitute(values) for x in config['overrides']]
    # Replace existing Hydra keys, including + extension keys, instead of duplicating them.
    for item in extra:
        if '=' not in item:
            raise ValueError('Each --override must be KEY=VALUE')
        key = item.split('=', 1)[0].lstrip('+')
        matches = [i for i, x in enumerate(overrides) if x.split('=', 1)[0].lstrip('+') == key]
        if matches:
            i = matches[0]
            original_key = overrides[i].split('=', 1)[0]
            overrides[i] = original_key + '=' + item.split('=', 1)[1]
        else:
            overrides.append(item)
    return [sys.executable, '-m', config['entrypoint'], *overrides]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', choices=['hs', 'math'], required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--train-data', type=Path, required=True)
    p.add_argument('--valid-data', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=ROOT)
    p.add_argument('--run-name', required=True)
    p.add_argument('--useful-model', type=Path)
    p.add_argument('--harmless-model', type=Path)
    p.add_argument('--calibration', type=Path)
    p.add_argument('--override', action='append', default=[])
    p.add_argument('--dry-run', action='store_true', help='Print command; no model/GPU/verl needed')
    p.add_argument('--resolve', action='store_true', help='Resolve Hydra config without training; requires verl')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', a.run_name):
        p.error('run-name must contain only letters, digits, _, . or -')
    values = {'CODE_ROOT': str(ROOT), 'POLICY_MODEL': str(a.model.resolve()),
              'TRAIN_DATA': str(a.train_data.resolve()), 'VALID_DATA': str(a.valid_data.resolve()),
              'OUTPUT_DIR': str(a.output_dir.resolve()), 'RUN_NAME': a.run_name}
    for arg, key in [('useful_model', 'USEFUL_MODEL'), ('harmless_model', 'HARMLESS_MODEL'), ('calibration', 'CALIBRATION')]:
        value = getattr(a, arg)
        if a.scenario == 'hs' and value is None: p.error('--' + arg.replace('_', '-') + ' is required for hs')
        if value is not None: values[key] = str(value.resolve())
    # Hydra treats spaces/commas in bare path overrides as syntax; quote all deployment paths.
    # CODE_ROOT is embedded in a longer path, so quote the final RHS below instead.
    command = build_command(a.scenario, values, a.override)
    for i in range(3, len(command)):
        key, value = command[i].split('=', 1)
        if value.startswith('/'):
            command[i] = key + '=' + json.dumps(value)
    if a.dry_run:
        print(json.dumps(command, indent=2)); return
    for name in ['model','train_data','valid_data'] + (['useful_model','harmless_model','calibration'] if a.scenario=='hs' else []):
        if not getattr(a,name).exists(): p.error(f'Missing {name}: {getattr(a,name)}')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), str(Path(env.get('VERL_ROOT', ROOT / 'third_party/verl')).resolve()), env.get('PYTHONPATH','')])
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    if a.resolve:
        subprocess.run(command[:3] + ['--cfg','job','--resolve'] + command[3:], env=env, check=True); return
    run = a.output_dir.resolve() / 'outputs' / a.run_name
    checkpoint = a.output_dir.resolve() / 'checkpoints' / a.run_name
    if checkpoint.exists(): p.error(f'Checkpoint directory already exists: {checkpoint}')
    run.mkdir(parents=True, exist_ok=False)
    (run/'launch.json').write_text(json.dumps({'scenario':a.scenario,'command':command},indent=2)+'\n')
    # Per-run writable paths; keep inherited HOME and authentication untouched.
    for key, sub in [('RAY_TMPDIR','ray'),('TMPDIR','tmp'),('TRITON_CACHE_DIR','triton'),('VLLM_CACHE_ROOT','vllm')]:
        path = run / sub;path.mkdir();env[key]=str(path)
    os.execvpe(sys.executable, command, env)

if __name__ == '__main__': main()
