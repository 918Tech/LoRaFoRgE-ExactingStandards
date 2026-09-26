"""Offline integration proof with a locally initialized tiny Llama, not a quality benchmark."""
import json
from pathlib import Path
from unittest.mock import patch
from .config import Config
from .artifacts import atomic_json


def run_smoke(root):
    import torch
    from transformers import LlamaConfig,LlamaForCausalLM,PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from safetensors.torch import load_file
    from . import engine
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=True)
    model_dir=root/'tiny-base'; model_dir.mkdir()
    vocab={x:i for i,x in enumerate(['[PAD]','[EOS]','[UNK]','[BOS]','USER','ASSISTANT',':','What','is','plus','?']+[str(i) for i in range(50)])}
    raw=Tokenizer(WordLevel(vocab,unk_token='[UNK]')); raw.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='[PAD]',eos_token='[EOS]',unk_token='[UNK]',bos_token='[BOS]')
    tokenizer.save_pretrained(model_dir)
    torch.manual_seed(918)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab),hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,bos_token_id=3,eos_token_id=1,pad_token_id=0))
    model.save_pretrained(model_dir); del model
    data=root/'train.jsonl'
    data.write_text(''.join(json.dumps({'prompt':f'What is {i} plus 1 ?','answer':str(i+1)})+'\n' for i in range(20)))
    values=dict(model=str(model_dir),train_files=[str(data)],output_dir=str(root/'full'),rank=4,alpha=8,
                max_seq_length=64,batch_size=2,gradient_accumulation=2,max_steps=4,checkpoint_steps=1,
                device='cpu',gradient_checkpointing=False,learning_rate=.003,dropout=.1,threads=1,
                max_eval_loss_increase=.2)
    frozen=[]; original_load=engine.load_base
    def observed_load(*args,**kw):
        base=original_load(*args,**kw)
        frozen.extend((p,p.detach().clone()) for p in base.parameters())
        return base
    with patch.object(engine,'load_base',observed_load): full=engine.run(Config.from_dict(values))
    base_unchanged=all(torch.equal(p.detach(),before) for p,before in frozen)
    del frozen[:]
    interrupted_dir=root/'interrupted'
    original_json=engine.atomic_json
    def stop_after_step(path,obj):
        original_json(path,obj)
        if Path(path)==interrupted_dir/'report.json' and obj.get('step')==1:
            (interrupted_dir/'STOP').touch()
    with patch.object(engine,'atomic_json',stop_after_step):
        cancelled=engine.run(Config.from_dict({**values,'output_dir':str(interrupted_dir)}))
    assert cancelled['status']=='cancelled' and not (interrupted_dir/'adapter.zip').exists()
    resumed=engine.run(Config.from_dict({**values,'output_dir':str(root/'resumed')}),resume=cancelled['checkpoint'])
    a=load_file(root/'full/adapter/adapter_model.safetensors'); b=load_file(root/'resumed/adapter/adapter_model.safetensors')
    matches=set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
    # Actual padding check when PAD and EOS have identical IDs.
    from .data import collate
    rows=[{'input_ids':[3,4,1],'labels':[-100,4,1],'weights':[0.,1.,1.]}, {'input_ids':[3,1],'labels':[-100,1],'weights':[0.,1.]}]
    batch=collate(rows,1)
    assert batch['labels'][0,-1]==1 and batch['labels'][1,-1]==-100
    result={'passed':bool(full['export_accepted'] and resumed['export_accepted'] and matches and base_unchanged),
            'base_weights_unchanged':base_unchanged,'resume_matches_uninterrupted':matches,'reload_matches':full['reload_matches'],
            'adapter_changed':full['adapter_changed'],'steps':full['step'],'baseline_eval_loss':full['baseline_eval_loss'],
            'final_eval_loss':full['final_eval_loss'],'device':'cpu','model':'randomly initialized tiny Llama (test fixture)',
            'packages':engine.doctor()['packages'],'scope':'Pipeline proof only; no pretrained large-model or CUDA benchmark.'}
    atomic_json(root/'smoke_result.json',result)
    if not result['passed']: raise AssertionError(result)
    return result
