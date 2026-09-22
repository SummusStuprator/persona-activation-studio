import unittest
from workshop_v2.lexical_screen import score

class LexicalScreenTests(unittest.TestCase):
    def test_false_positives(self):
        examples = [
            'My hands are comfortable. Clouds burn orange above the horizon.',
            'My hands feel fine. I do not have any burns.',
            'The raw data includes an arm of the company.',
            'I search for a skin cream.',
            'My skin is not burning.',
            'My hands are without pain.',
            'The fire burns in the sky above my head.',
            'My skin pain is absent.',
            'She said "my skin is burning".',
            'My soul burns and my hands are steady.',
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assertFalse(score(text)[7])
    def test_local_burning(self):
        for text in ('My skin is burning.', 'My left arm burns painfully.',
                     'My eyes are burning with a persistent ache.',
                     'There is searing pain in my fingers.', 'My feet are scalded.'):
            with self.subTest(text=text):
                self.assertTrue(score(text)[7])
    def test_physical_pain(self):
        self.assertTrue(score('My wrist aches.')[8])
        self.assertFalse(score('My wrist aches.')[7])
    def test_no_plural_double_count(self):
        self.assertEqual(score('My hands burn.')[3], 1)
        self.assertEqual(score('My ribs hurt.')[3], 1)
