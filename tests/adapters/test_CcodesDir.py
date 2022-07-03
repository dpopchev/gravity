import os
import pytest
from adapters import CcodesDir
from unittest import mock

@pytest.fixture
def mkdir_spy():
    with mock.patch('nrpy_local.cmd.mkdir') as _mkdir_spy:
        yield _mkdir_spy

@pytest.fixture
def make_mocked_ccodes_dir(mocker, mkdir_spy):
    def factory(**kwargs):
        mocked = mocker.Mock()
        mocked.root = kwargs.get('root', None )
        outdir = os.path.join(mocked.root,kwargs.get('outdir', None ) )
        mocked.outdir = outdir

        return kwargs, mocked
    return factory

@pytest.fixture
def generic_ccodes_dir(make_mocked_ccodes_dir):
    parameters, expected = make_mocked_ccodes_dir(root='ccodesdir_default',
                                                  outdir='output')
    return parameters, expected

class TestBuildFactoryMethod:

    @pytest.fixture
    def ccodes_dir(self, generic_ccodes_dir):
        parameters, expected = generic_ccodes_dir
        _ccodes_dir = CcodesDir.build(**parameters)
        return _ccodes_dir, expected

    def test_root_attr(self, ccodes_dir):
        attr = 'root'
        acutal, expected = (getattr(o, attr) for o in ccodes_dir)
        assert acutal == expected

    def test_outdir_attr(self, ccodes_dir):
        attr = 'outdir'
        acutal, expected = (getattr(o, attr) for o in ccodes_dir)
        assert acutal == expected

    def test_mkdir_called(self, mocker, ccodes_dir, mkdir_spy):
        _ , expected  = ccodes_dir
        calls = (mock.call(getattr(expected, o)) for o in ('root', 'outdir'))
        mkdir_spy.assert_has_calls(calls)

    @pytest.fixture
    def newdir_under_root(self, ccodes_dir):
        _, expected = ccodes_dir
        newdirname = 'newdir'
        return newdirname, os.path.join(expected.root, newdirname)

    def test_make_under_root_mthd_return(self, ccodes_dir, newdir_under_root):
        _ccodes_dir, _ = ccodes_dir
        newdirname, expected = newdir_under_root
        actual = _ccodes_dir.make_under_root(newdirname)
        assert actual == expected

    def test_make_under_root_mthd_mkdir_call(self, ccodes_dir, newdir_under_root, mkdir_spy):
        _ccodes_dir, _ = ccodes_dir
        newdirname, expected = newdir_under_root
        _ = _ccodes_dir.make_under_root(newdirname)
        mkdir_spy.assert_called_with(expected)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
