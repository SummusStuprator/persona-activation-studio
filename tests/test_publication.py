"""Release archive checks use synthetic file names and content."""
from pathlib import Path
import importlib.util
import unittest
import tempfile
import warnings
import zipfile
import subprocess
from contextlib import redirect_stdout
import io
from unittest.mock import patch
from studio_paths import ASSET_ROOT

spec = importlib.util.spec_from_file_location('release_audit', ASSET_ROOT / 'scripts/verify_release.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class PublicationTests(unittest.TestCase):
    def test_research_fixture_is_allowed(self):
        audit.validate_entry('paper/datasets/example.json', b'{"fixture": true}')

    def test_generated_files_are_rejected(self):
        for name in ('runs/sample.json', 'models/example.gguf', 'account.db',
                     'studio.toml', 'paper_reproduction/result.json', '.env',
                     '.env.production', 'backup.sqlite3', 'x_accounts.db-journal',
                     'reports/private.json', 'cache/input.json', 'private.pem'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.validate_entry(name, b'fixture')

    def test_unsafe_paths_are_rejected(self):
        for name in ('../secret.txt', '/absolute/file.txt', 'backups/private.txt',
                     'C:/temp/file.txt', 'a/./file.txt', 'a//file.txt', ''):
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

    def test_session_cookie_pattern_is_rejected(self):
        with self.assertRaises(ValueError):
            audit.validate_entry('example.txt', ('auth_' + 'token=' + 'a' * 40).encode())

    def test_duplicate_archive_members_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'duplicate.whl'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                with zipfile.ZipFile(path, 'w') as archive:
                    archive.writestr('example.txt', b'first')
                    archive.writestr('example.txt', b'second')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                audit.archive_entries(path)

    def test_missing_runtime_module_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing packaged'):
            audit.verify_source_payload({}, ['studio_cli.py'], wheel=True)

    def test_missing_nested_document_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing packaged'):
            audit.verify_source_payload({}, ['docs/releases/0.3.4.md'], wheel=True)

    def test_verification_rejects_dirty_checkout_before_reading_artifacts(self):
        with patch.object(audit, 'git', return_value=' M README.md'):
            with self.assertRaisesRegex(ValueError, 'Commit source changes'):
                audit.verify(Path('unused'))

    def test_history_checks_clean_blobs_and_reused_private_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                subprocess.run(['git', '-C', str(root), '-c', 'commit.gpgsign=false',
                    '-c', 'core.hooksPath=' + str(root / 'no-hooks'), *args],
                    check=True, capture_output=True)
            git('init', '-q')
            git('config', 'user.name', 'Release audit fixture')
            git('config', 'user.email', 'fixture@example.invalid')
            (root / 'example.txt').write_text('synthetic fixture')
            git('add', 'example.txt')
            git('commit', '-qm', 'Public fixture')
            with patch.object(audit, 'ROOT', root), redirect_stdout(io.StringIO()):
                audit.verify_history()
            # The same blob under a different name must still trigger the gate,
            # even after the private filename has been removed from HEAD.
            (root / '.env').write_text('synthetic fixture')
            git('add', '.env')
            git('commit', '-qm', 'Historical private path')
            git('rm', '.env')
            git('commit', '-qm', 'Remove historical path')
            with patch.object(audit, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'Private/runtime file'):
                    audit.verify_history()


if __name__ == '__main__':
    unittest.main()
