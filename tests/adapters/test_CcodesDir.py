import pytest
from adapters import CcodesDir

class TestEmptyCcodesDir:
    @pytest.fixture
    def ccodes_dir(self):
        _ccodes_dir = CcodesDir()
        return _ccodes_dir

    def test_root_attr(self, ccodes_dir):
        expected = None
        acutal = ccodes_dir.root
        assert acutal is expected

    def test_outdir_attr(self, ccodes_dir):
        expected = None
        acutal = ccodes_dir.outdir
        assert acutal is expected

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
