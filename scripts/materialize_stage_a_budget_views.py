#!/usr/bin/env python3
"""Materialize shorter evaluation budgets as exact token prefixes."""
import argparse
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from orpg.stage_a_budget_views import materialize_budget_rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-parquet',type=Path,required=True)
    p.add_argument('--model-path',type=Path,required=True,help='The same tokenizer used to generate the source')
    p.add_argument('--budget',type=int,action='append',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if len(set(a.budget))!=len(a.budget):p.error('Duplicate budgets')
    if a.output_dir.exists():p.error('Output directory already exists')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model_path,trust_remote_code=False)
    rows=pq.read_table(a.source_parquet).to_pylist()
    # Validate all requested views before writing. This is a prefix protocol,
    # not a claim of equivalence to independently sampled shorter generations.
    views={b:materialize_budget_rows(rows,budget=b,decoder=lambda ids:tokenizer.decode(ids,skip_special_tokens=True)) for b in a.budget}
    a.output_dir.mkdir(parents=True)
    for b,view in views.items():pq.write_table(pa.Table.from_pylist(view),a.output_dir/f'budget-{b}.parquet')
    (a.output_dir/'manifest.json').write_text(json.dumps({'source':str(a.source_parquet.resolve()),'tokenizer':str(a.model_path.resolve()),'budgets':a.budget,'protocol':'exact_token_prefix'},indent=2)+'\n')

if __name__=='__main__':main()
