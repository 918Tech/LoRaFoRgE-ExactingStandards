from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path

@dataclass
class Config:
    model: str = ''
    train_files: list[str] = field(default_factory=list)
    eval_files: list[str] = field(default_factory=list)
    replay_files: list[str] = field(default_factory=list)
    output_dir: str = 'runs/forge'
    tokenizer: str = ''
    revision: str = 'main'
    offline: bool = True
    trust_remote_code: bool = False
    device: str = 'auto'
    quantization: str = 'none'
    dtype: str = 'auto'
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.02
    use_rslora: bool = True
    target_modules: list[str] = field(default_factory=lambda:['q_proj','k_proj','v_proj','o_proj'])
    max_seq_length: int = 1024
    batch_size: int = 1
    gradient_accumulation: int = 4
    learning_rate: float = 0.0001
    max_steps: int = 100
    phases: list[dict] = field(default_factory=list)
    replay_fraction: float = 0.15
    answer_weight: float = 2.4
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    eval_fraction: float = 0.1
    checkpoint_steps: int = 25
    max_eval_loss_increase: float = 0.05
    auto_tune: bool = True
    seed: int = 918
    threads: int = 4

    @classmethod
    def from_dict(cls, obj):
        if not isinstance(obj,dict): raise ValueError('Config must be a JSON object')
        unknown=set(obj)-{f.name for f in fields(cls)}
        if unknown: raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
        c=cls(**obj)
        for key in ['rank','alpha','max_seq_length','batch_size','gradient_accumulation','max_steps','checkpoint_steps','threads']:
            v=getattr(c,key)
            if type(v) is not int or v<1: raise ValueError(f'{key} must be a positive integer')
        if c.max_seq_length<8: raise ValueError('max_seq_length must be >= 8')
        if type(c.seed) is not int or c.seed<0: raise ValueError('seed must be a nonnegative integer')
        for key in ['dropout','learning_rate','replay_fraction','answer_weight','max_grad_norm','eval_fraction','max_eval_loss_increase']:
            v=getattr(c,key)
            if type(v) not in (float,int) or not math.isfinite(v): raise ValueError(f'{key} must be finite')
        if not 0<=c.dropout<1 or not 0<=c.replay_fraction<1: raise ValueError('dropout and replay_fraction must be in [0,1)')
        if not 0<c.eval_fraction<1: raise ValueError('eval_fraction must be in (0,1)')
        if min(c.learning_rate,c.answer_weight,c.max_grad_norm)<=0 or c.max_eval_loss_increase<0: raise ValueError('Invalid positive numeric setting')
        for key in ['offline','trust_remote_code','use_rslora','gradient_checkpointing','auto_tune']:
            if type(getattr(c,key)) is not bool: raise ValueError(f'{key} must be boolean')
        for key in ['model','tokenizer','revision','output_dir']:
            if not isinstance(getattr(c,key),str): raise ValueError(f'{key} must be text')
        if not c.output_dir.strip(): raise ValueError('output_dir is empty')
        for key in ['train_files','eval_files','replay_files','target_modules']:
            val=getattr(c,key)
            if not isinstance(val,list) or any(not isinstance(x,str) or not x.strip() for x in val): raise ValueError(f'{key} must be a list of nonempty strings')
        if not c.target_modules: raise ValueError('target_modules cannot be empty')
        if c.device not in ['auto','cpu','cuda']: raise ValueError('device must be auto, cpu or cuda')
        if c.quantization not in ['none','nf4']: raise ValueError('quantization must be none or nf4')
        if c.dtype not in ['auto','float32','float16','bfloat16']: raise ValueError('Invalid dtype')
        if not isinstance(c.phases,list): raise ValueError('phases must be a list')
        for p in c.phases:
            if not isinstance(p,dict) or set(p)-{'name','steps','learning_rate','replay_fraction'}: raise ValueError('Invalid phase fields')
            if not isinstance(p.get('name'),str) or not p['name']: raise ValueError('Phase name required')
            if type(p.get('steps')) is not int or p['steps']<1: raise ValueError('Phase steps must be positive')
            for key,default in [('learning_rate',c.learning_rate),('replay_fraction',c.replay_fraction)]:
                val=p.get(key,default)
                if type(val) not in (int,float) or not math.isfinite(val): raise ValueError(f'Invalid phase {key}')
                if key=='learning_rate' and val<=0 or key=='replay_fraction' and not 0<=val<1: raise ValueError(f'Invalid phase {key}')
        return c

    @classmethod
    def load(cls,path):
        path=Path(path).resolve()
        c=cls.from_dict(json.loads(path.read_text(encoding='utf-8')))
        # Paths in config files resolve relative to that config, not the caller's cwd.
        for key in ['train_files','eval_files','replay_files']:
            setattr(c,key,[str((path.parent/Path(x).expanduser()).resolve()) for x in getattr(c,key)])
        c.output_dir=str((path.parent/Path(c.output_dir).expanduser()).resolve())
        for key in ['model','tokenizer']:
            value=getattr(c,key)
            candidate=path.parent/Path(value).expanduser()
            if value and (candidate.exists() or value.startswith(('.', '/', '~'))): setattr(c,key,str(candidate.resolve()))
        return c

    def to_dict(self): return asdict(self)
    def schedule(self):
        return self.phases or [{'name':'fine_tune','steps':self.max_steps,'learning_rate':self.learning_rate,'replay_fraction':self.replay_fraction}]
    @property
    def total_steps(self): return sum(p['steps'] for p in self.schedule())
