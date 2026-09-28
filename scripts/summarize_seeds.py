#!/usr/bin/env python3
"""Aggregate explicitly labeled evaluation seeds for one fixed checkpoint."""
import argparse
import json
import math
from pathlib import Path
import statistics

def aggregate(records, metrics):
    if len(records)<2 or len({seed for seed,_ in records})!=len(records):
        raise ValueError('At least two distinct evaluation seeds are required')
    result={}
    for key in metrics:
        values=[]
        for _,record in records:
            value=record
            for part in key.split('.'): value=value[part]
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
                raise ValueError(f'Nonfinite/non-numeric metric: {key}')
            values.append(float(value))
        result[key]={'avg':statistics.fmean(values),'sample_std':statistics.stdev(values),'per_seed':dict(zip((str(s) for s,_ in records),values))}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',action='append',required=True,help='SEED=summary.json (repeat)')
    p.add_argument('--metric',action='append',required=True,help='Dot-separated numeric field')
    p.add_argument('--checkpoint',required=True,help='Fixed checkpoint identifier shared by these evaluations')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();records=[];sources={}
    for item in a.input:
        seed,path=item.split('=',1);seed=int(seed);path=Path(path).resolve()
        record=json.loads(path.read_text())
        if 'seed' in record and record['seed']!=seed: p.error('Input seed differs from summary seed')
        if str(path) in sources.values(): p.error('Repeated input file')
        records.append((seed,record));sources[str(seed)]=str(path)
    out={'checkpoint':a.checkpoint,'variation':'evaluation_seed','ddof':1,'sources':sources,'metrics':aggregate(records,a.metric)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
