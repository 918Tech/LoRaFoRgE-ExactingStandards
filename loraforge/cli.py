import argparse
import json
from pathlib import Path
from .config import Config


def main(argv=None):
    p=argparse.ArgumentParser(prog='loraforge',description='918 Technologies standalone LoRA fine-tuner')
    p.add_argument('--version',action='version',version='LoRA Forge 1.0.0')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor',help='Report runtime and GPU availability')
    init=sub.add_parser('init',help='Write a starter configuration'); init.add_argument('path')
    for name in ['validate','train']:
        s=sub.add_parser(name); s.add_argument('config')
        if name=='train': s.add_argument('--resume',help='Checkpoint directory; select a new output_dir in config')
    serve=sub.add_parser('serve',help='Open local browser dashboard'); serve.add_argument('--port',type=int,default=9180); serve.add_argument('--no-browser',action='store_true')
    smoke=sub.add_parser('smoke',help='Real offline tiny-model training, export and resume test'); smoke.add_argument('--output',default='runs/smoke')
    audit=sub.add_parser('audit',help='Validate saved PEFT adapter tensor pairs and rank'); audit.add_argument('directory')
    args=p.parse_args(argv)
    try:
        if args.command=='init':
            path=Path(args.path)
            with path.open('x',encoding='utf-8') as f: json.dump(Config().to_dict(),f,indent=2)
            print(f'Created {path}; set model and train_files before training.');return 0
        if args.command=='doctor':
            from .engine import doctor
            print(json.dumps(doctor(),indent=2));return 0
        if args.command=='validate':
            from .engine import input_preflight
            print(json.dumps(input_preflight(Config.load(args.config)),indent=2));return 0
        if args.command=='train':
            from .engine import run
            result=run(Config.load(args.config),args.resume)
            return 0 if result['status']=='completed' else 2
        if args.command=='serve':
            from .server import serve
            serve(args.port,not args.no_browser);return 0
        if args.command=='smoke':
            from .smoke import run_smoke
            print(json.dumps(run_smoke(Path(args.output)),indent=2));return 0
        if args.command=='audit':
            from .artifacts import audit_adapter
            print(json.dumps(audit_adapter(args.directory),indent=2));return 0
    except Exception as exc:
        p.exit(1,f'LoRA Forge: {type(exc).__name__}: {exc}\n')

if __name__=='__main__': raise SystemExit(main())
