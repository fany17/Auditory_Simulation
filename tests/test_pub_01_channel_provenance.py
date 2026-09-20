import gzip
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('channel',Path(__file__).parents[1]/'scripts/pub_01_sparrkulee_channel_provenance.py')
channel=importlib.util.module_from_spec(spec);spec.loader.exec_module(channel)


class HeaderTests(unittest.TestCase):
    def fixture(self,n=64,rate=1024):
        root=Path(__file__).parents[1]/'test_fixtures';root.mkdir(exist_ok=True)
        path=Path(tempfile.mkdtemp(prefix='header_',dir=root))/'header.bdf.gz'
        h=bytearray(b' '*(256*(n+1)));h[:8]=b'\xffBIOSEMI'
        def field(pos,width,value):
            h[pos:pos+width]=str(value).encode().ljust(width)
        for pos,width,value in [(184,8,len(h)),(236,8,1),(244,8,1),(252,4,n)]:
            field(pos,width,value)
        for i in range(n):
            for offset,width,value in [(0,16,f'A{i+1}'),(96,8,'uV'),(104,8,-100),(112,8,100),(120,8,-8388608),(128,8,8388607),(216,8,rate)]:
                field(256+offset*n+i*width,width,value)
        with gzip.open(path,'wb') as f:f.write(h)
        return path

    def test_exact_64_channels(self):
        result=channel.header(self.fixture())
        self.assertEqual(len(result['channel_names']),64)
        self.assertEqual(result['physical_units'],['uV']*64)
        self.assertTrue(all(result['calibration_valid']))

    def test_short_channel_dimension_refused(self):
        with self.assertRaises(ValueError):channel.header(self.fixture(n=63))

    def test_zero_rate_refused(self):
        with self.assertRaises(ValueError):channel.header(self.fixture(rate=0))


if __name__=='__main__':unittest.main()
