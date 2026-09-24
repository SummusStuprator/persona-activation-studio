from types import SimpleNamespace
from unittest.mock import patch
import unittest
from persona.handles import normalized_handle
import studio_cli


class HandleTests(unittest.TestCase):
    def test_handle_normalization(self):
        self.assertEqual(normalized_handle(' @Example_1 '), 'Example_1')

    def test_paths_urls_queries_and_device_names_are_rejected(self):
        for value in ('../outside', 'a/b', 'a\\b', 'https://x.com/example',
                      'user OR from:other', '', '@', '@@user', 'a' * 16, 'CON', 'lpt1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalized_handle(value)

    def test_invalid_handle_fails_before_workspace_or_scraper_is_started(self):
        with patch.object(studio_cli, 'ensure_workspace') as workspace, patch.object(studio_cli, 'run_module') as run:
            with self.assertRaises(ValueError):
                studio_cli.scrape_cmd(SimpleNamespace(handle=['../outside']))
            workspace.assert_not_called()
            run.assert_not_called()
