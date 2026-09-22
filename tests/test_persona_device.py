import json
import tempfile
import unittest
from pathlib import Path
from workshop_v2.persona_device import GIB, select_device


class PersonaDeviceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        config = dict(hidden_size=2560, num_hidden_layers=36,
                      num_attention_heads=32, num_key_value_heads=8,
                      head_dim=128, vocab_size=151936)
        (root / 'config.json').write_text(json.dumps(config))
        self.model = dict(base_path=str(root), base_bytes=8*GIB,
                          adapter_bytes=128*2**20, files=[])

    def choose(self, mode='Auto', precision='auto', **kwargs):
        options = dict(cuda_available=True, vram_free=8*GIB,
                       vram_total=8*GIB, nf4_available=True)
        options.update(kwargs)
        return select_device(self.model, 512, mode, precision, **options)

    def test_auto_selects_nf4_when_bf16_does_not_fit(self):
        result = self.choose()
        self.assertEqual((result['device'], result['precision']), ('cuda:0', 'nf4'))

    def test_auto_selects_bf16_when_it_fits(self):
        self.assertEqual(self.choose(vram_free=24*GIB, vram_total=24*GIB)['precision'], 'bf16')

    def test_auto_cpu_fallback(self):
        self.assertEqual(self.choose(cuda_available=False)['device'], 'cpu')
        self.assertEqual(self.choose(vram_free=GIB)['device'], 'cpu')

    def test_explicit_cuda_never_falls_back_to_cpu(self):
        with self.assertRaises(MemoryError):
            self.choose('CUDA', vram_free=GIB)
        with self.assertRaises(RuntimeError):
            self.choose('CUDA', cuda_available=False)

    def test_explicit_bf16_never_quantizes(self):
        with self.assertRaises(MemoryError):
            self.choose('CUDA', 'bf16')

    def test_nf4_dependency_and_cpu_restrictions(self):
        with self.assertRaises(RuntimeError):
            self.choose('CUDA', 'nf4', nf4_available=False)
        with self.assertRaises(ValueError):
            self.choose('CPU', 'nf4')

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            self.choose('typo')
