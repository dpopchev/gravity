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

class TestMakeUnderRootMethod:

    new_dirname = 'newdir'

    @pytest.fixture
    def mkdir_spy(self, mocker):
        with mock.patch('nrpy_local.cmd.mkdir') as _mkdir_spy:
            yield _mkdir_spy

    @pytest.fixture
    def ccodes_dir(self, mkdir_spy):
        _ccodes_dir = CcodesDir.build()
        return _ccodes_dir

    @pytest.fixture
    def expected_defaults(self):
        root = 'ccodesdir_default'
        outdir = 'ccodesdir_default/output'
        _expected = CcodesDirStub(root, outdir)
        return _expected

    @pytest.fixture
    def make_under_root(self, ccodes_dir, expected_defaults):
        expected_new_dir = os.path.join(expected_defaults.root, self.new_dirname)
        actual = ccodes_dir.make_under_root(self.new_dirname)
        return actual

    def test_make_under_root_return(self, make_under_root, expected_defaults ):
        expected = os.path.join(expected_defaults.root, self.new_dirname)
        assert make_under_root == expected

    def test_make_under_root_mkdir_call(self, make_under_root, expected_defaults, mkdir_spy):
        expected_call = os.path.join(expected_defaults.root, self.new_dirname)
        mkdir_spy.assert_any_call(expected_call)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
