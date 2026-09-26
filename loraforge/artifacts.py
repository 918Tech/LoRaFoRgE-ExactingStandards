import hashlib
import json
import os
import zipfile
from pathlib import Path


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def adapter_hash(model):
    from peft import get_peft_model_state_dict
    h=hashlib.sha256()
    for name,t in sorted(get_peft_model_state_dict(model).items()):
        h.update(name.encode()); h.update(t.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def audit_adapter(path):
    import torch
    from safetensors import safe_open
    path=Path(path)
    cfg=json.loads((path/'adapter_config.json').read_text())
    if cfg.get('peft_type')!='LORA' or type(cfg.get('r')) is not int or cfg['r']<1: raise ValueError('Invalid LoRA config/rank')
    if cfg.get('rank_pattern'): raise ValueError('Variable rank adapters are not supported by this exporter')
    with safe_open(path/'adapter_model.safetensors',framework='pt',device='cpu') as f:
        keys=set(f.keys())
        if not keys: raise ValueError('Empty adapter tensor file')
        pairs=0
        for key in sorted(keys):
            t=f.get_tensor(key)
            if not torch.isfinite(t).all(): raise ValueError(f'Nonfinite adapter tensor: {key}')
            if '.lora_A.' in key:
                other=key.replace('.lora_A.','.lora_B.')
                if other not in keys: raise ValueError(f'Missing adapter pair: {key}')
                b=f.get_tensor(other)
                if t.ndim!=2 or b.ndim!=2 or t.shape[0]!=cfg['r'] or b.shape[1]!=cfg['r']: raise ValueError(f'Adapter rank mismatch: {key}')
                pairs+=1
            elif '.lora_B.' in key:
                if key.replace('.lora_B.','.lora_A.') not in keys: raise ValueError(f'Orphan adapter tensor: {key}')
            else: raise ValueError(f'Unexpected non-LoRA tensor: {key}')
        if not pairs: raise ValueError('No paired LoRA tensors')
    return {'rank':cfg['r'],'pairs':pairs,'tensors':len(keys),'sha256':file_hash(path/'adapter_model.safetensors')}


def export_adapter(path,destination):
    path=Path(path); destination=Path(destination)
    audit=audit_adapter(path)
    tmp=destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['adapter_config.json','adapter_model.safetensors']: z.write(path/name,name)
    os.replace(tmp,destination)
    return {**audit,'zip_sha256':file_hash(destination),'zip_bytes':destination.stat().st_size}
