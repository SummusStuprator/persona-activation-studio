"""Release archive checks use synthetic file names and content."""
from pathlib import Path
import importlib.util
import unittest
from studio_paths import ASSET_ROOT

spec = importlib.util.spec_from_file_location('release_audit', ASSET_ROOT / 'scripts/verify_release.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class PublicationTests(unittest.TestCase):
    def test_research_fixture_is_allowed(self):
        audit.validate_entry('paper/datasets/example.json', b'{"fixture": true}')

    def test_generated_files_are_rejected(self):
        for name in ('runs/sample.json', 'models/example.gguf', 'account.db',
                     'studio.toml', 'paper_reproduction/result.json', '.env'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.validate_entry(name, b'fixture')

    def test_unsafe_paths_are_rejected(self):
        for name in ('../secret.txt', '/absolute/file.txt', 'backups/private.txt'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.validate_entry(name, b'fixture')

    def test_credential_patterns_are_rejected(self):
        with self.assertRaises(ValueError):
            audit.validate_entry('example.txt', ('gh' + 'p_' + 'x' * 40).encode())

    def test_workstation_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            audit.validate_entry('example.txt', ('/ho' + 'me/example/private').encode())

    def test_source_line_endings_are_portable(self):
        self.assertTrue(audit.source_matches(b'one\ntwo\n', b'one\r\ntwo\r\n'))

    def test_different_source_content_is_rejected(self):
        self.assertFalse(audit.source_matches(b'one\ntwo\n', b'one\r\nthree\r\n'))

    def test_documentation_can_name_configuration_files(self):
        audit.validate_entry('README.md', b'Configure studio.toml and run studio check.')


if __name__ == '__main__':
    unittest.main()
