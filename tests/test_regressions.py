import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

class RegressionTests(unittest.TestCase):
    def test_confidence_affects_single_example_gradient(self):
        import torch
        from loraforge.engine import weighted_loss
        from loraforge.data import collate
        class Fixed(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.logits=torch.nn.Parameter(torch.tensor([[[0.,1.,2.],[0.,1.,2.],[0.,1.,2.]]]))
            def forward(self,**kw): return type('Output',(),{'logits':self.logits})()
        gradients=[]
        for confidence in [1.,.1]:
            model=Fixed()
            batch=collate([{'input_ids':[0,1,2],'labels':[-100,1,2],'weights':[0.,confidence,confidence],'confidence':confidence}],0)
            loss,_=weighted_loss(model,batch);loss.backward();gradients.append(model.logits.grad.clone())
        self.assertTrue(torch.allclose(gradients[1],gradients[0]*.1))
    def test_tokenizer_normalizer_changes_identity(self):
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer,normalizers
        from tokenizers.models import WordLevel
        from loraforge.config import Config
        from loraforge.engine import model_identity
        raw=Tokenizer(WordLevel({'[UNK]':0,'hello':1} ,unk_token='[UNK]'))
        tok=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'weights.bin').write_bytes(b'fixture')
            c=Config(model=d)
            before=model_identity(c,None,tok)
            tok.backend_tokenizer.normalizer=normalizers.Lowercase()
            self.assertNotEqual(before,model_identity(c,None,tok))
    def test_dead_worker_reports_failure(self):
        from loraforge.server import App
        with tempfile.TemporaryDirectory() as d:
            app=App(Path(d)/'jobs'); app.out=Path(d)/'output';app.out.mkdir()
            (app.out/'report.json').write_text(json.dumps({'status':'training','step':3}))
            app.process=subprocess.Popen([sys.executable,'-c','raise SystemExit(7)']);app.process.wait()
            state=app.state()
            self.assertFalse(state['running']);self.assertEqual(state['report']['status'],'failed')
            self.assertIn('7',state['report']['error'])
    def test_late_stop_refuses_export(self):
        import loraforge.engine as engine
        from loraforge.smoke import run_smoke
        from unittest.mock import patch
        original=engine.atomic_json
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'full/report.json'
            def late_stop(p,obj):
                original(p,obj)
                if Path(p)==path and obj.get('step')==4: (path.parent/'STOP').touch()
            with patch.object(engine,'atomic_json',late_stop):
                try: run_smoke(Path(d))
                except (KeyError,AssertionError,FileNotFoundError): pass
            status=json.loads(path.read_text())
            self.assertEqual(status['status'],'cancelled')
            self.assertFalse((path.parent/'adapter.zip').exists())

if __name__=='__main__': unittest.main()
