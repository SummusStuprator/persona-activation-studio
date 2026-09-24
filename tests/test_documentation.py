"""Resolve user-facing Markdown file links in source and installed wheels."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
import re
import unittest
from studio_paths import ASSET_ROOT


class DocumentationTests(unittest.TestCase):
    def test_local_documentation_links_resolve(self):
        files = [ASSET_ROOT / name for name in ('README.md', 'SECURITY.md', 'CONTRIBUTING.md', 'THIRD_PARTY.md')]
        files += list((ASSET_ROOT / 'docs').rglob('*.md'))
        self.assertGreater(len(files), 10)
        for source in files:
            content = source.read_text(encoding='utf-8')
            for raw in re.findall(r'\[[^\]\n]+\]\(([^\s)]+)\)', content):
                url = urlsplit(raw)
                if url.scheme or url.netloc or not url.path:
                    continue
                target = source.parent / unquote(url.path)
                with self.subTest(document=source.name, link=raw):
                    self.assertTrue(target.exists(), f'Missing link target: {raw}')
