import os
import pytest
from adapters import CcodesDir
from collections import namedtuple

CcodesDirStub = namedtuple('CcodesDirStub',  'root outdir')

class TestEmptyConstruction:
    @pytest.fixture
    def ccodes_dir(self):
        _ccodes_dir = CcodesDir()
        return _ccodes_dir

    @pytest.fixture
    def expected(self):
        _expected = CcodesDirStub(None, None)
        return _expected

    def test_root_attr(self, ccodes_dir, expected):
        attr = 'root'
        acutal = getattr(ccodes_dir, attr)
        assert acutal is getattr(expected, attr)

    def test_outdir_attr(self, ccodes_dir, expected):
        attr = 'outdir'
        acutal = getattr(ccodes_dir, attr)
        assert acutal is getattr(expected, attr)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
