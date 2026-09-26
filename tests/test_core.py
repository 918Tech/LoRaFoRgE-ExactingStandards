import json
import tempfile
import unittest
from pathlib import Path

try:
    from loraforge.config import Config
    from loraforge.data import load_rows, split_rows, encode_row, collate
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

class Tokenizer:
    eos_token_id = 2
    pad_token_id = 2
    bos_token_id = 1
    chat_template = None
    def encode(self, text, add_special_tokens=False):
        return [ord(x)+3 for x in text]

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, 'Standalone LoRA Forge core must exist')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
    def rows(self, rows):
        p = self.root/'data.jsonl'
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return load_rows([str(p)])
    def test_unknown_config_rejected(self):
        with self.assertRaises(ValueError): Config.from_dict({'learing_rate': 1})
    def test_invalid_numeric_values_rejected(self):
        for key, val in [('rank',0),('learning_rate',float('nan')),('eval_fraction',1),('max_seq_length',2),('seed',True)]:
            with self.subTest(key=key), self.assertRaises(ValueError): Config.from_dict({key:val})
    def test_empty_data_rejected(self):
        with self.assertRaises(ValueError): self.rows([])
    def test_answer_zero_and_duplicate(self):
        rows = self.rows([{'prompt':'p','answer':0},{'prompt':'p','answer':0}])
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['answer'],'0')
    def test_conflicting_labels_rejected(self):
        with self.assertRaises(ValueError): self.rows([{'prompt':'p','answer':'a'},{'prompt':'p','answer':'b'}])
    def test_group_split_no_leakage(self):
        rows=self.rows([{'prompt':f'p{i}','answer':str(i),'group_id':str(i//2)} for i in range(12)])
        train, val=split_rows(rows, .25, 918)
        self.assertTrue(train and val)
        self.assertFalse({r['group_id'] for r in train}&{r['group_id'] for r in val})
    def test_prompt_and_answer_masks(self):
        row=self.rows([{'prompt':'problem','answer':'42','reasoning':'because'}])[0]
        enc=encode_row(row,Tokenizer(),128,answer_weight=3)
        self.assertTrue(all(x==-100 for x in enc['labels'][:enc['prompt_length']]))
        self.assertEqual(enc['labels'][-1],2)
        self.assertEqual(enc['weights'][-2],3)
        self.assertEqual(enc['labels'][enc['prompt_length']:],enc['input_ids'][enc['prompt_length']:])
    def test_overlong_input_never_silently_cropped(self):
        row=self.rows([{'prompt':'x'*100,'answer':'42'}])[0]
        with self.assertRaisesRegex(ValueError,'sequence'): encode_row(row,Tokenizer(),32)
    def test_chat_final_assistant_only(self):
        rows=self.rows([{'messages':[{'role':'user','content':'hello'},{'role':'assistant','content':'hi'}]}])
        self.assertEqual(rows[0]['answer'],'hi')
        with self.assertRaises(ValueError): self.rows([{'messages':[{'role':'user','content':'hello'}]}])
    def test_input_output_alias(self):
        row=self.rows([{'instruction':'Add','input':'1+1','output':'2'}])[0]
        self.assertIn('1+1',row['prompt'])
        self.assertEqual(row['answer'],'2')

if __name__=='__main__': unittest.main()
