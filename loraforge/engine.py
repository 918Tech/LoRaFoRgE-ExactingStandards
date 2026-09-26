"""Single-device real LoRA/QLoRA training with fail-closed exports."""
import gc
import importlib.metadata
import json
import math
import os
import random
import shutil
import sqlite3
import time
import tempfile
from pathlib import Path

from .artifacts import adapter_hash, atomic_json, audit_adapter, export_adapter, file_hash
from .data import collate, digest, encode_row, prepare_data


def doctor():
    result={'python':__import__('sys').version.split()[0],'packages':{},'cuda':False,'devices':[]}
    for name in ['torch','transformers','peft','accelerate','safetensors','bitsandbytes']:
        try: result['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result['packages'][name]=None
    if result['packages']['torch']:
        import torch
        result['cuda']=torch.cuda.is_available()
        result['devices']=[{'name':torch.cuda.get_device_name(i),'vram_gib':round(torch.cuda.get_device_properties(i).total_memory/2**30,2)} for i in range(torch.cuda.device_count())]
    return result


def input_preflight(c):
    if not c.model.strip(): raise ValueError('Set model to a local Hugging Face model directory or model ID')
    if c.model.lower().endswith('.gguf'): raise ValueError('GGUF is not a training checkpoint; use original Hugging Face model weights')
    train,val,replay=prepare_data(c)
    return {'train_rows':len(train),'eval_rows':len(val),'replay_rows':len(replay),
            'total_steps':c.total_steps,'input_fingerprint':digest([train,val,replay]),
            'runtime':doctor(),'note':'Input preflight only. train performs tokenizer, sequence, model and target validation before optimization.'}


def load_base(c, device, dtype):
    from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
    kwargs=dict(revision=c.revision,local_files_only=c.offline,trust_remote_code=c.trust_remote_code)
    config=AutoConfig.from_pretrained(c.model,**kwargs)
    if getattr(config,'quantization_config',None):
        raise ValueError('Prequantized base checkpoints are not supported in v1; select original BF16/FP16 weights and optionally quantization=nf4')
    kwargs.update(dtype=dtype,low_cpu_mem_usage=True)
    if c.quantization=='nf4':
        if device!='cuda': raise ValueError('NF4 QLoRA requires a CUDA GPU in this release')
        kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype),device_map={'':0})
    model=AutoModelForCausalLM.from_pretrained(c.model,**kwargs)
    if c.quantization=='none': model.to(device)
    model.config.use_cache=False
    return model


def weighted_loss(model,batch,weighted=True):
    import torch
    logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).logits[:,:-1]
    labels=batch['labels'][:,1:]; active=labels!=-100
    if not active.any(): raise ValueError('Batch has no supervised tokens')
    losses=torch.nn.functional.cross_entropy(logits[active].float(),labels[active],reduction='none')
    weights=batch['weights'][:,1:][active] if weighted else torch.ones_like(losses)
    # Normalize token emphasis independently from record confidence.
    # Otherwise singleton microbatches cancel confidence completely.
    confidence=batch['confidence'][:,None].expand_as(labels)[active] if weighted else torch.ones_like(weights)
    return (losses*weights).sum()/(weights/confidence).sum(), int(active.sum())


def evaluate(model,rows,pad,device,batch_size):
    import torch
    model.eval(); loss_sum=0.0; tokens=0
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            loss,n=weighted_loss(model,collate(rows[start:start+batch_size],pad,device),weighted=False)
            loss_sum+=float(loss)*n; tokens+=n
    value=loss_sum/tokens
    if not math.isfinite(value): raise ValueError('Nonfinite validation loss')
    return value


def phase_at(c,step):
    end=0
    for p in c.schedule():
        end+=p['steps']
        if step<end: return p
    return c.schedule()[-1]


def model_identity(c,base,tokenizer):
    p=Path(c.model)
    if p.is_dir():
        assets=sorted(x for x in p.rglob('*') if x.is_file() and x.suffix in {'.json','.safetensors','.bin','.py'})
        weights=[x for x in assets if x.suffix in {'.safetensors','.bin'}]
        if not weights: raise ValueError('No base model weight files found')
        identity={str(x.relative_to(p)):file_hash(x) for x in assets}
    else:
        commit=getattr(base.config,'_commit_hash',None)
        if not commit: raise ValueError('Cannot resolve immutable model revision; download a local model snapshot first')
        identity={'model':c.model,'commit':commit}
    with tempfile.TemporaryDirectory(prefix='forge-tokenizer-identity-') as tmp:
        tokenizer.save_pretrained(tmp)
        tokenizer_assets={str(x.relative_to(tmp)):file_hash(x) for x in sorted(Path(tmp).rglob('*')) if x.is_file()}
    return digest({'model':identity,'tokenizer_assets':tokenizer_assets})


def run(c,resume=None):
    import torch
    from peft import LoraConfig,PeftModel,get_peft_model,prepare_model_for_kbit_training,get_peft_model_state_dict
    from transformers import AutoTokenizer,set_seed

    if int(os.environ.get('WORLD_SIZE','1'))!=1: raise ValueError('v1 is single-device; run python -m loraforge directly')
    out=Path(c.output_dir).resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('output_dir must be empty/new; resume uses a NEW output_dir and the existing checkpoint as input')
    out.mkdir(parents=True,exist_ok=True)
    started=time.time(); model=base=optimizer=None; step=0
    status={'status':'preparing','actual_training_complete':False,'export_accepted':False,'step':0,'total_steps':c.total_steps}
    def emit(event,**data):
        row={'time':time.time(),'event':event,**data}
        with (out/'events.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    def report(**kw):
        status.update(kw); atomic_json(out/'report.json',status)
    def checkpoint():
        final=out/f'checkpoint-{step:06d}'
        if final.exists(): return final
        tmp=out/f'.checkpoint-{step:06d}.tmp'; tmp.mkdir()
        model.save_pretrained(tmp,safe_serialization=True)
        state={'step':step,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
               'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else [],
               'fingerprint':fingerprint,'initial_hash':initial_hash,'baseline_eval_loss':baseline,
               'best_eval_loss':best,'lr_scale':lr_scale,'last_eval_loss':last_eval}
        torch.save(state,tmp/'training_state.pt')
        atomic_json(tmp/'config.json',c.to_dict())
        audit_adapter(tmp); tmp.rename(final)
        report(checkpoint=str(final),step=step)
        emit('checkpoint',step=step,path=str(final))
        return final
    def cancelled():
        if not (out/'STOP').exists(): return False
        checkpoint()
        (out/'adapter.zip').unlink(missing_ok=True)
        report(status='cancelled',export_accepted=False,step=step)
        emit('cancelled',step=step)
        return True
    try:
        atomic_json(out/'config.json',c.to_dict()); report()
        info=input_preflight(c); emit('inputs',**info)
        torch.set_num_threads(c.threads); set_seed(c.seed)
        device=('cuda' if torch.cuda.is_available() else 'cpu') if c.device=='auto' else c.device
        if device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable')
        if c.dtype=='auto': dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float32
        else: dtype=getattr(torch,c.dtype)
        if device=='cpu' and dtype!=torch.float32: raise ValueError('Use float32 on CPU in this release')
        if dtype==torch.bfloat16 and device=='cuda' and not torch.cuda.is_bf16_supported(): raise ValueError('GPU does not support bfloat16')
        tokenizer=AutoTokenizer.from_pretrained(c.tokenizer or c.model,revision=c.revision,local_files_only=c.offline,trust_remote_code=c.trust_remote_code)
        if tokenizer.eos_token_id is None: raise ValueError('Tokenizer requires an EOS token')
        if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
        raw_train,raw_eval,raw_replay=prepare_data(c)
        encoded=[[encode_row(r,tokenizer,c.max_seq_length,c.answer_weight) for r in rows] for rows in [raw_train,raw_eval,raw_replay]]
        train,val,replay=encoded
        with sqlite3.connect(out/'memory.sqlite') as db:
            db.execute('CREATE TABLE examples (prompt_id TEXT PRIMARY KEY, split TEXT, category TEXT, source TEXT, content_hash TEXT)')
            for split,rows in zip(['train','eval','replay'],[raw_train,raw_eval,raw_replay]):
                db.executemany('INSERT INTO examples VALUES (?,?,?,?,?)',[(r['prompt_id'],split,r['category'],r['source'],digest(r)) for r in rows])
        emit('sequences_validated',max_tokens=max(len(r['input_ids']) for rows in encoded for r in rows),train_rows=len(train),eval_rows=len(val),replay_rows=len(replay))
        report(status='loading_model'); base=load_base(c,device,dtype)
        config_limit=getattr(base.config,'max_position_embeddings',None)
        if isinstance(config_limit,int) and max(len(r['input_ids']) for rows in encoded for r in rows)>config_limit: raise ValueError('Dataset exceeds model position limit')
        identity=model_identity(c,base,tokenizer)
        semantic=c.to_dict(); semantic.pop('output_dir')
        fingerprint=digest({'config':semantic,'data':info['input_fingerprint'],'encoded':digest(encoded),'model':identity,'packages':info['runtime']['packages'],'device':device,'dtype':str(dtype)})
        state=None
        if resume:
            resume=Path(resume).resolve(); audit_adapter(resume)
            state=torch.load(resume/'training_state.pt',map_location='cpu',weights_only=True)
            if state['fingerprint']!=fingerprint: raise ValueError('Resume identity mismatch: config, model, data, runtime, or device changed')
        # Target only supported projection layers and reject partially mistyped selectors.
        targets=[]
        from transformers.pytorch_utils import Conv1D
        for selector in c.target_modules:
            matches=[name for name,m in base.named_modules() if (name==selector or name.endswith('.'+selector)) and isinstance(m,(torch.nn.Linear,Conv1D))]
            if not matches: raise ValueError(f'No supported LoRA projection matches {selector!r}; inspect model module names')
            targets+=matches
        targets=sorted(set(targets))
        if any(name.endswith(('lm_head','embed_tokens')) for name in targets): raise ValueError('Select internal projections, not embeddings/output head')
        if c.quantization=='nf4': base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=c.gradient_checkpointing)
        if resume: model=PeftModel.from_pretrained(base,str(resume),is_trainable=True)
        else: model=get_peft_model(base,LoraConfig(r=c.rank,lora_alpha=c.alpha,lora_dropout=c.dropout,use_rslora=c.use_rslora,target_modules=targets,bias='none',task_type='CAUSAL_LM'))
        if c.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            model.enable_input_require_grads()
        trainable=[p for p in model.parameters() if p.requires_grad]
        if not trainable: raise ValueError('No trainable adapter parameters')
        if any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad): raise ValueError('Unexpected trainable base weights')
        optimizer=torch.optim.AdamW(trainable,lr=c.learning_rate)
        scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda' and dtype==torch.float16)
        initial_hash=adapter_hash(model); lr_scale=1.0
        report(status='evaluating',model_identity=identity,fingerprint=fingerprint,trainable_parameters=sum(p.numel() for p in trainable),targets=targets,device=device,dtype=str(dtype))
        if state:
            step=state['step']
            if not 0<=step<=c.total_steps: raise ValueError('Invalid checkpoint step')
            optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
            initial_hash=state['initial_hash']; baseline=state['baseline_eval_loss']; best=state['best_eval_loss']; lr_scale=state['lr_scale']; last_eval=state['last_eval_loss']
            torch.set_rng_state(state['torch_rng'])
            if device=='cuda': torch.cuda.set_rng_state_all(state['cuda_rng'])
            emit('resumed',step=step,checkpoint=str(resume))
        else:
            baseline=evaluate(model,val,tokenizer.pad_token_id,device,c.batch_size); best=baseline; last_eval=baseline
            emit('baseline',eval_loss=baseline)
        report(status='training',step=step,baseline_eval_loss=baseline)
        while step<c.total_steps:
            if cancelled(): return status
            phase=phase_at(c,step); rng=random.Random(c.seed+step)
            lr=phase.get('learning_rate',c.learning_rate)*lr_scale
            for group in optimizer.param_groups: group['lr']=lr
            model.train(); optimizer.zero_grad(set_to_none=True); total_loss=0.0
            for micro in range(c.gradient_accumulation):
                samples=[]
                for _ in range(c.batch_size):
                    pool=replay if replay and rng.random()<phase.get('replay_fraction',c.replay_fraction) else train
                    samples.append(rng.choice(pool))
                loss,_=weighted_loss(model,collate(samples,tokenizer.pad_token_id,device))
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss; export refused')
                total_loss+=float(loss.detach())/c.gradient_accumulation
                scaler.scale(loss/c.gradient_accumulation).backward()
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(trainable,c.max_grad_norm,error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update(); step+=1
            emit('step',step=step,total_steps=c.total_steps,phase=phase['name'],loss=total_loss,learning_rate=lr,grad_norm=float(norm))
            report(step=step,loss=total_loss,phase=phase['name'],elapsed_seconds=time.time()-started)
            phase_done=step==c.total_steps or phase_at(c,step)['name']!=phase['name']
            if step%c.checkpoint_steps==0 or phase_done:
                last_eval=evaluate(model,val,tokenizer.pad_token_id,device,c.batch_size)
                if c.auto_tune and last_eval>best:
                    lr_scale=max(0.0625,lr_scale*0.5)
                    emit('auto_tune',reason='heldout_loss_regression',lr_scale=lr_scale)
                best=min(best,last_eval)
                emit('evaluation',step=step,eval_loss=last_eval,best_eval_loss=best)
                checkpoint()
        if cancelled(): return status
        final_hash=adapter_hash(model)
        report(actual_training_complete=True,step=step,baseline_eval_loss=baseline,final_eval_loss=last_eval,adapter_changed=final_hash!=initial_hash)
        if initial_hash==final_hash: raise ValueError('Adapter parameters did not change; export refused')
        if last_eval>baseline*(1+c.max_eval_loss_increase): raise ValueError('Heldout loss regression exceeds export gate; trained checkpoints remain available')
        report(status='verifying')
        # Save then reload the actual adapter onto the same frozen base. Compare both
        # tensors and probe logits without loading a second large base model.
        stage=out/'.adapter.tmp'; stage.mkdir()
        model.save_pretrained(stage,safe_serialization=True); audit=audit_adapter(stage)
        expected={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
        model.eval(); probe=collate(val[:1],tokenizer.pad_token_id,device)
        with torch.no_grad(): before=model(input_ids=probe['input_ids'],attention_mask=probe['attention_mask']).logits.detach().cpu()
        del optimizer; optimizer=None
        unwrapped=model.unload(); del model; model=None
        # PEFT unload removes wrappers but leaves this metadata attribute.
        if hasattr(unwrapped,'peft_config'): delattr(unwrapped,'peft_config')
        model=PeftModel.from_pretrained(unwrapped,str(stage),is_trainable=False); model.eval()
        actual=get_peft_model_state_dict(model)
        if set(expected)!=set(actual) or any(not torch.equal(v,actual[k].detach().cpu().to(v.dtype)) for k,v in expected.items()): raise ValueError('Reloaded adapter tensor mismatch')
        with torch.no_grad(): after=model(input_ids=probe['input_ids'],attention_mask=probe['attention_mask']).logits.detach().cpu()
        if not torch.allclose(before.float(),after.float(),atol=1e-4,rtol=1e-4): raise ValueError('Reloaded adapter output mismatch')
        if cancelled(): return status
        stage.rename(out/'adapter'); tokenizer.save_pretrained(out/'tokenizer')
        package=export_adapter(out/'adapter',out/'adapter.zip')
        if cancelled(): return status
        report(status='completed',export_accepted=True,reload_matches=True,package=package,elapsed_seconds=time.time()-started,
               note='Validation loss is a local heldout measurement, not a competition score or accuracy guarantee.')
        emit('completed',step=step,eval_loss=last_eval,adapter_zip=str(out/'adapter.zip'))
        return status
    except BaseException as exc:
        # Any exception after optimization invalidates the success artifact.
        (out/'adapter.zip').unlink(missing_ok=True)
        report(status='failed',export_accepted=False,step=step,error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.time()-started)
        emit('failed',step=step,error=str(exc)); raise
    finally:
        optimizer=None; model=None; base=None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
