import os
import pytest
from adapters import CcodesDir
from collections import namedtuple
from unittest import mock

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
        acutal, _expected = map(lambda _: getattr(_, attr), (ccodes_dir, expected))
        assert acutal is _expected

    def test_outdir_attr(self, ccodes_dir, expected):
        attr = 'outdir'
        acutal, _expected = map(lambda _: getattr(_, attr), (ccodes_dir, expected))
        assert acutal is _expected

class TestBuildFactoryMethod:
    @pytest.fixture
    def mkdir_spy(self, mocker):
        with mock.patch('nrpy_local.cmd.mkdir') as _mkdir_spy:
            yield _mkdir_spy

    @pytest.fixture
    def ccodes_dir(self, mkdir_spy):
        _ccodes_dir = CcodesDir.build()
        return _ccodes_dir

    @pytest.fixture
    def expected(self):
        root = 'ccodesdir_default'
        outdir = 'ccodesdir_default/output'
        _expected = CcodesDirStub(root, outdir)
        return _expected

    def test_root_attr(self, ccodes_dir, expected):
        attr = 'root'
        acutal, _expected = map(lambda _: getattr(_, attr), (ccodes_dir, expected))
        assert acutal is _expected

    def test_outdir_attr(self, ccodes_dir, expected):
        attr = 'outdir'
        acutal, _expected = map(lambda _: getattr(_, attr), (ccodes_dir, expected))
        assert acutal == _expected

    def test_mkdir_called(self, ccodes_dir, expected, mkdir_spy):
        calls = (mock.call(_) for _ in (expected.root, expected.outdir) )
        mkdir_spy.assert_has_calls(calls)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
