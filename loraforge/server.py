"""Loopback dashboard. No shell execution, public binding or remote mutation."""
import json
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from .config import Config
from .artifacts import atomic_json


def tail(path,limit=100000):
    if not path.is_file(): return ''
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-limit)); return f.read(limit).decode('utf-8',errors='replace')


class App:
    def __init__(self,work):
        self.work=Path(work); self.work.mkdir(parents=True,exist_ok=True)
        self.token=secrets.token_urlsafe(32); self.lock=threading.Lock()
        self.process=None; self.out=None; self.log=None; self.log_handle=None
    def state(self):
        with self.lock:
            running=self.process is not None and self.process.poll() is None
            status={}; events=[]
            if self.out:
                try: status=json.loads((self.out/'report.json').read_text())
                except (FileNotFoundError,json.JSONDecodeError): pass
                for line in tail(self.out/'events.jsonl').splitlines():
                    try: events.append(json.loads(line))
                    except json.JSONDecodeError: pass
            exit_code=self.process.poll() if self.process else None
            if self.process and not running and (status.get('status') not in {'completed','cancelled','failed'} or (exit_code != 0 and status.get('status')=='completed')):
                status={**status,'status':'failed','export_accepted':False,'error':f'Worker exited unexpectedly (exit code {exit_code}). Inspect the run console; use the last saved checkpoint.'}
                if self.out and self.out.is_dir(): atomic_json(self.out/'report.json',status)
            return {'running':running,'report':status,'events':events[-250:],
                    'exit_code':self.process.poll() if self.process else None,'log':tail(self.log) if self.log else '',
                    'cwd':str(Path.cwd()),'download':bool(not running and status.get('export_accepted') and self.out and (self.out/'adapter.zip').is_file())}
    def start(self,raw,resume=None):
        with self.lock:
            if self.process and self.process.poll() is None: raise ValueError('A training run is already active')
            c=Config.from_dict(raw)
            for key in ['train_files','eval_files','replay_files']:
                setattr(c,key,[str(Path(x).expanduser().resolve()) for x in getattr(c,key)])
            for key in ['model','tokenizer']:
                val=getattr(c,key)
                if val and (Path(val).expanduser().exists() or val.startswith(('.', '/', '~'))): setattr(c,key,str(Path(val).expanduser().resolve()))
            c.output_dir=str(Path(c.output_dir).expanduser().resolve())
            out=Path(c.output_dir)
            if out.exists() and any(out.iterdir()): raise ValueError('Choose an empty/new output directory')
            if not c.model or not c.train_files: raise ValueError('Model and dataset paths are required')
            job=self.work/secrets.token_hex(6);job.mkdir()
            config=job/'config.json'; atomic_json(config,c.to_dict())
            if self.log_handle: self.log_handle.close()
            self.out=out; self.log=job/'process.log'; self.log_handle=self.log.open('w',encoding='utf-8')
            cmd=[sys.executable,'-u','-m','loraforge','train',str(config)]
            if resume: cmd+=['--resume',str(Path(resume).expanduser().resolve())]
            env=os.environ.copy()
            package_root=str(Path(__file__).resolve().parent.parent)
            env['PYTHONPATH']=package_root+os.pathsep+env.get('PYTHONPATH','')
            self.process=subprocess.Popen(cmd,stdout=self.log_handle,stderr=subprocess.STDOUT,env=env)
            return {'started':True,'pid':self.process.pid,'output_dir':str(out)}
    def stop(self):
        with self.lock:
            if not self.process or self.process.poll() is not None: raise ValueError('No active run')
            if self.out.is_dir(): (self.out/'STOP').touch()
            else: self.process.terminate()
            return {'stop_requested':True}
    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired: self.process.kill();self.process.wait()
        if self.log_handle: self.log_handle.close()


def make_server(port=9180,work=None):
    app=App(work or Path(tempfile.mkdtemp(prefix='loraforge-')))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def allowed(self):
            host=self.headers.get('Host','')
            allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
            origin=self.headers.get('Origin')
            return host in allowed and (origin is None or origin in {'http://'+h for h in allowed})
        def respond(self,status,body,kind='application/json'):
            if isinstance(body,(dict,list)): body=json.dumps(body).encode()
            elif isinstance(body,str): body=body.encode()
            self.send_response(status); self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('X-Frame-Options','DENY');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if not self.allowed(): return self.respond(403,{'error':'Local access only'})
            path=urlsplit(self.path).path
            if path=='/':
                html=(Path(__file__).parent/'web/index.html').read_text(encoding='utf-8').replace('__TOKEN__',app.token)
                return self.respond(200,html,'text/html; charset=utf-8')
            if path=='/api/state': return self.respond(200,app.state())
            if path=='/api/defaults': return self.respond(200,Config().to_dict())
            if path=='/api/doctor':
                from .engine import doctor
                return self.respond(200,doctor())
            if path=='/api/download':
                state=app.state()
                if not state['download']: return self.respond(404,{'error':'No verified export available'})
                p=app.out/'adapter.zip'
                self.send_response(200);self.send_header('Content-Type','application/zip');self.send_header('Content-Disposition','attachment; filename="LoRA-Forge-adapter.zip"');self.send_header('Content-Length',str(p.stat().st_size));self.end_headers()
                with p.open('rb') as f:
                    while chunk:=f.read(1024*1024): self.wfile.write(chunk)
                return
            return self.respond(404,{'error':'Not found'})
        def do_POST(self):
            if not self.allowed() or not secrets.compare_digest(self.headers.get('X-Forge-Token',''),app.token): return self.respond(403,{'error':'Local session token required'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=1048576: raise ValueError('Expected JSON body, maximum 1 MiB')
                obj=json.loads(self.rfile.read(length))
                if not isinstance(obj,dict): raise ValueError('Expected JSON object')
                if self.path=='/api/start': result=app.start(obj.get('config'),obj.get('resume'))
                elif self.path=='/api/stop': result=app.stop()
                elif self.path=='/api/validate':
                    from .engine import input_preflight
                    result=input_preflight(Config.from_dict(obj.get('config')))
                else: return self.respond(404,{'error':'Not found'})
                self.respond(200,result)
            except Exception as exc: self.respond(400,{'error':f'{type(exc).__name__}: {exc}'})
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.app=app
    return server


def serve(port,open_browser=True):
    with tempfile.TemporaryDirectory(prefix='loraforge-') as tmp:
        server=make_server(port,Path(tmp)); url=f'http://127.0.0.1:{server.server_port}'
        print(f'LoRA Forge → {url}\nLeave this terminal open. Press Ctrl+C to exit.',flush=True)
        if open_browser: webbrowser.open(url)
        try: server.serve_forever()
        except KeyboardInterrupt: pass
        finally: server.app.close();server.server_close()
