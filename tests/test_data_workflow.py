"""Exercise the documented local-export workflow with invented conversations."""
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import csv
import io
import json
import unittest
from persona.dataset_builder import build


class LocalExportTests(unittest.TestCase):
    def test_documented_exports_build_disjoint_conversation_splits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / 'project'
            project.mkdir()
            exports = root / 'exports'
            (exports / 'reply_context').mkdir(parents=True)
            rows, parents = [], []
            for index in range(40):
                topic = 'topic' + chr(97 + index // 26) + chr(97 + index % 26)
                conversation = str(1000 + index)
                parents.append(dict(id=conversation, username='demo_reader', kind='tweet',
                    date=f'2026-01-{1 + index % 28:02d}T12:00:00Z', conversation_id=conversation,
                    text=f'How would you investigate {topic} in a new project?'))
                rows.append(dict(id=str(2000 + index), username='demo_author', kind='reply',
                    date=f'2026-01-{1 + index % 28:02d}T12:01:00Z', conversation_id=conversation,
                    in_reply_to_tweet_id=conversation, in_reply_to_username='demo_reader',
                    text=f'I would write down the assumptions about {topic} and test each one carefully.'))
            with (exports / 'all_posts.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (exports / 'reply_context/parents.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in parents), encoding='utf-8')
            output = root / 'revision'
            args = SimpleNamespace(input=project, exports=exports, output=output,
                overwrite=False, max_context_posts=8, train_ratio=.8, validation_ratio=.1)
            with redirect_stdout(io.StringIO()):
                build(args)
            splits = []
            for split in ('train', 'validation', 'test'):
                path = output / f'profiles/demo_author/sft/portable/core/{split}.jsonl'
                records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
                self.assertTrue(records, split)
                splits.append({row['id'] for row in records})
            self.assertEqual(sum(map(len, splits)), 40)
            self.assertFalse(splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
            self.assertTrue((output / 'manifest.json').is_file())
