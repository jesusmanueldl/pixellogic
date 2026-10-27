import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import studio

class TestPixelLogicAndQR(unittest.TestCase):
    def test_phrase_round_trip(self):
        phrase='Laboratorio de lo Imposible'
        with tempfile.TemporaryDirectory() as d:
            meta=studio.create_pair(phrase,d,pixels=900,level='H')
            self.assertTrue(meta['pixel'].exists())
            for key,reader in [('pixel','PixelLogic E35'),('qr','QR OpenCV')]:
                results=studio.read_both(meta[key])
                hit=next(r for r in results if r['sistema']==reader)
                self.assertTrue(hit['ok'],(key,results))
                self.assertEqual(hit['texto'],phrase)

    def test_unicode_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            msg='¡Hola, Tabasco! 2026'
            meta=studio.create_pair(msg,d,pixels=900,level='Q')
            result=studio.read_both(meta['pixel'])
            self.assertEqual(result[0]['texto'],msg)

    def test_missing_image_is_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises((ValueError,FileNotFoundError)):
                studio.read_both(Path(d)/'missing.png')

if __name__=='__main__':unittest.main()
