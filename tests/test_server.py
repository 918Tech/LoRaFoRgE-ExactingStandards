import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError

class ServerTests(unittest.TestCase):
    def test_local_dashboard_boundaries(self):
        self.assertIsNotNone(importlib.util.find_spec('loraforge.server'),'Local dashboard must exist')
        from loraforge.server import make_server
        with tempfile.TemporaryDirectory() as tmp:
            server=make_server(0,Path(tmp)); thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
            url=f'http://127.0.0.1:{server.server_port}'
            with urlopen(url) as r:
                self.assertIn('LoRA Forge',r.read().decode())
            req=Request(url+'/api/validate',data=b'{}',headers={'Content-Type':'application/json'},method='POST')
            with self.assertRaises(HTTPError) as caught: urlopen(req)
            self.assertEqual(caught.exception.code,403)
            with self.assertRaises(HTTPError) as caught: urlopen(Request(url,headers={'Host':'evil.example'}))
            self.assertEqual(caught.exception.code,403)
            with self.assertRaises(HTTPError) as caught: urlopen(Request(url+'/api/validate',data=b'{}',headers={'Content-Type':'application/json','X-Forge-Token':server.app.token,'Origin':'https://evil.example'},method='POST'))
            self.assertEqual(caught.exception.code,403)
            with urlopen(url+'/api/state') as r: self.assertEqual(json.load(r)['running'],False)
            with self.assertRaises(HTTPError) as caught: urlopen(url+'/api/download')
            self.assertEqual(caught.exception.code,404)

if __name__=='__main__': unittest.main()
