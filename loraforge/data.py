"""Strict corpus ingestion; no implicit synthetic fallback or lossy truncation."""
import csv
import hashlib
import json
import math
import random
import unicodedata
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def normalized(text): return ' '.join(unicodedata.normalize('NFKC',text).split())


def first(row, keys, default=None):
    return next((row[k] for k in keys if k in row and row[k] is not None),default)


def normalize(row, source):
    if not isinstance(row,dict): raise ValueError(f'{source}: record must be an object')
    messages=row.get('messages')
    if messages is not None:
        if not isinstance(messages,list) or len(messages)<2: raise ValueError(f'{source}: chat needs context and final assistant')
        if any(not isinstance(m,dict) or m.get('role') not in ['system','user','assistant'] or not isinstance(m.get('content'),str) for m in messages): raise ValueError(f'{source}: invalid chat messages')
        if messages[-1]['role']!='assistant': raise ValueError(f'{source}: last message must be assistant')
        prompt=json.dumps(messages[:-1],ensure_ascii=False,sort_keys=True)
        answer=messages[-1]['content']
        messages=messages[:-1]
        reasoning=''
    else:
        prompt=first(row,['prompt','instruction','question','problem'])
        answer=first(row,['answer','completion','output','response','target'])
        if prompt is None or answer is None: raise ValueError(f'{source}: prompt and answer fields required')
        prompt=str(prompt)
        if row.get('instruction') is not None and row.get('input') is not None: prompt+='\n'+str(row['input'])
        reasoning=str(row.get('reasoning') or '')
    answer=str(answer)
    if not str(prompt).strip() or not answer.strip(): raise ValueError(f'{source}: empty prompt/answer')
    weight=row.get('weight',1.0)
    try: weight=float(weight)
    except (TypeError,ValueError): raise ValueError(f'{source}: invalid weight')
    if not math.isfinite(weight) or weight<=0: raise ValueError(f'{source}: weight must be positive and finite')
    return dict(prompt=str(prompt),answer=answer,reasoning=reasoning,messages=messages,
                category=str(row.get('category') or 'general'),weight=weight,source=source,
                prompt_id=digest(normalized(str(prompt))),group_id=str(row.get('group_id') or digest(normalized(str(prompt)))))


def load_rows(paths):
    result=[]; seen={}
    if not paths: raise ValueError('No dataset files supplied')
    for name in paths:
        p=Path(name)
        if not p.is_file(): raise ValueError(f'Dataset missing: {p}')
        if p.suffix.lower()=='.jsonl':
            with p.open(encoding='utf-8-sig') as f:
                records=[json.loads(s) for s in f if s.strip()]
        elif p.suffix.lower()=='.json':
            records=json.loads(p.read_text(encoding='utf-8-sig'))
            if isinstance(records,dict): records=first(records,['data','records','train','examples','rows'])
        elif p.suffix.lower()=='.csv':
            with p.open(encoding='utf-8-sig',newline='') as f: records=list(csv.DictReader(f))
        else: raise ValueError(f'Unsupported dataset extension: {p.suffix}')
        if not isinstance(records,list): raise ValueError(f'{p}: expected a list of records')
        for i,raw in enumerate(records):
            r=normalize(raw,f'{p}:{i+1}'); key=r['prompt_id']
            if key in seen:
                if seen[key]['answer']!=r['answer']: raise ValueError(f'Conflicting answers for prompt at {r["source"]}')
                continue
            seen[key]=r; result.append(r)
    if not result: raise ValueError('Dataset is empty')
    return result


def split_rows(rows, fraction, seed):
    groups={}
    for r in rows: groups.setdefault(r['group_id'],[]).append(r)
    keys=sorted(groups); random.Random(seed).shuffle(keys)
    if len(keys)<2: raise ValueError('At least two independent groups needed for train/validation split')
    n=max(1,min(len(keys)-1,round(len(keys)*fraction)))
    return [r for k in keys[n:] for r in groups[k]], [r for k in keys[:n] for r in groups[k]]


def disjoint(a,b):
    if {x['prompt_id'] for x in a}&{x['prompt_id'] for x in b} or {x['group_id'] for x in a}&{x['group_id'] for x in b}:
        raise ValueError('Data leakage: prompt or group overlaps validation/training/replay splits')


def prepare_data(config):
    rows=load_rows(config.train_files)
    if config.eval_files:
        train, val=rows,load_rows(config.eval_files); disjoint(train,val)
    else: train,val=split_rows(rows,config.eval_fraction,config.seed)
    replay=load_rows(config.replay_files) if config.replay_files else []
    if replay:
        disjoint(replay,val); disjoint(replay,train)
    return train,val,replay


def prompt_ids(row, tokenizer):
    messages=row.get('messages') or [{'role':'user','content':row['prompt']}]
    if getattr(tokenizer,'chat_template',None):
        return tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
    if row.get('messages'):
        text='\n'.join(f'{m["role"].upper()}:\n{m["content"]}' for m in messages)+'\nASSISTANT:\n'
    else: text='USER:\n'+row['prompt']+'\nASSISTANT:\n'
    ids=tokenizer.encode(text,add_special_tokens=False)
    if tokenizer.bos_token_id is not None: ids=[tokenizer.bos_token_id]+ids
    return ids


def encode_row(row,tokenizer,max_length,answer_weight=2.4):
    # Tokenize segments independently so prompt/answer boundaries cannot drift.
    prefix=prompt_ids(row,tokenizer)
    rationale=tokenizer.encode(row['reasoning']+'\n' if row['reasoning'] else '',add_special_tokens=False)
    answer=tokenizer.encode(row['answer'],add_special_tokens=False)
    if not prefix or not answer: raise ValueError(f'{row["source"]}: empty tokenized prompt or answer')
    eos=tokenizer.eos_token_id
    if eos is None: raise ValueError('Tokenizer needs an eos_token_id')
    target=rationale+answer+[eos]
    ids=prefix+target
    if len(ids)>max_length: raise ValueError(f'{row["source"]}: sequence length {len(ids)} exceeds {max_length}; increase max_seq_length or explicitly shorten the record')
    weights=[0.0]*len(prefix)+[row['weight']]*len(rationale)+[row['weight']*answer_weight]*(len(answer)+1)
    return dict(input_ids=ids,labels=[-100]*len(prefix)+target,weights=weights,prompt_length=len(prefix),confidence=row['weight'])


def collate(rows,pad_id,device='cpu'):
    import torch
    n=max(len(r['input_ids']) for r in rows)
    def padded(key,fill): return [r[key]+[fill]*(n-len(r[key])) for r in rows]
    return {'confidence':torch.tensor([r.get('confidence',1.0) for r in rows],device=device),
            'input_ids':torch.tensor(padded('input_ids',pad_id),device=device),
            'labels':torch.tensor(padded('labels',-100),device=device),
            'weights':torch.tensor(padded('weights',0.0),device=device),
            'attention_mask':torch.tensor([[1]*len(r['input_ids'])+[0]*(n-len(r['input_ids'])) for r in rows],device=device)}
