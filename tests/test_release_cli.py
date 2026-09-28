import importlib.util
from pathlib import Path
import statistics
import pytest

ROOT=Path(__file__).parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

@pytest.mark.parametrize('scenario,lr,aggregator',[('math','1e-06','pcgrad_priority'),('hs','2e-06','pcgrad')])
def test_portable_configuration_keeps_scientific_defaults(scenario,lr,aggregator):
    values={k:'/tmp/example' for k in ['CODE_ROOT','POLICY_MODEL','TRAIN_DATA','VALID_DATA','OUTPUT_DIR','USEFUL_MODEL','HARMLESS_MODEL','CALIBRATION']}
    values['RUN_NAME']='orpg-test'
    command=load('train').build_command(scenario,values)
    cfg=dict(x.split('=',1) for x in command[3:])
    assert cfg['trainer.total_training_steps']=='100'
    assert cfg['actor_rollout_ref.actor.optim.lr']==lr
    assert cfg['+actor_rollout_ref.actor.policy_loss.objective_wise_aggregator']==aggregator
    assert cfg['+actor_rollout_ref.actor.policy_loss.objective_wise_positive_rule']=='C'
    assert cfg['+actor_rollout_ref.actor.policy_loss.objective_wise_positive_q']=='0.5'
    assert cfg['+actor_rollout_ref.actor.policy_loss.objective_wise_positive_strength']=='0.25'
    assert cfg['data.seed']=='42'
    assert not any('${' in x for x in command)
    command=load('train').build_command(scenario,values,['actor_rollout_ref.actor.policy_loss.objective_wise_positive_strength=0'])
    assert '+actor_rollout_ref.actor.policy_loss.objective_wise_positive_strength=0' in command
    assert len([x for x in command if 'objective_wise_positive_strength=' in x])==1

def test_eval_seed_sample_std_and_invalid_inputs():
    agg=load('summarize_seeds').aggregate
    records=[(42,{'macro':{'score':1}}),(43,{'macro':{'score':2}}),(44,{'macro':{'score':6}})]
    result=agg(records,['macro.score'])['macro.score']
    assert result['avg']==3
    assert result['sample_std']==statistics.stdev([1,2,6])
    with pytest.raises(ValueError):agg([records[0],records[0]],['macro.score'])
    with pytest.raises(ValueError):agg([(42,{'x':1}),(43,{'x':float('nan')})],['x'])

def test_training_data_artifacts_are_written_without_content_checksums(tmp_path):
    from cw_grpo.math_stage_a_data import prepare_stage_a_records,write_stage_a_artifacts
    rows=[{'problem':f'Compute {i}+1.','answer':str(i+1)} for i in range(4)]
    prepared=prepare_stage_a_records(rows,tuning_dev_size=1,excluded_prompt_ids=())
    manifest=write_stage_a_artifacts(prepared,tmp_path,provenance={'source_revision':'fixture'})
    assert (tmp_path/'train.parquet').stat().st_size>0
    assert 'sha256' not in str(manifest.get('artifacts',{}))
