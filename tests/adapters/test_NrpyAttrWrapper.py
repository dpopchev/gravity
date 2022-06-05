import pytest

from adapters import NrpyAttrWrapper

class TestFunctionality:
    TESTCASE_NAME = 'Functionality test'
    TESTCASE_ARGS = ('arg1', 'arg2')
    TESTCASE_KWARGS = {'kwarg1': 'value', 'kwarg2': None}

    @pytest.fixture
    def callback(self, mocker):
        _callback = mocker.Mock()
        return _callback

    @pytest.fixture
    def nrpy_attr_wrapper(self, callback):
        _nrpy_attr_wrapper = NrpyAttrWrapper(
            self.TESTCASE_NAME, callback, self.TESTCASE_ARGS, self.TESTCASE_KWARGS
        )
        return _nrpy_attr_wrapper

    def test_name_attr(self, nrpy_attr_wrapper):
        attr = 'name'
        actual, expected = getattr(nrpy_attr_wrapper, attr), self.TESTCASE_NAME
        assert actual == expected

    def test_args_attr(self, nrpy_attr_wrapper):
        attr = 'args'
        actual, expected = getattr(nrpy_attr_wrapper, attr), self.TESTCASE_ARGS
        assert actual == expected

    def test_kwargs_attr(self, nrpy_attr_wrapper):
        attr = 'kwargs'
        actual, expected = getattr(nrpy_attr_wrapper, attr), self.TESTCASE_KWARGS
        assert actual == expected

    def test_callback_attr(self, nrpy_attr_wrapper, callback):
        attr = 'callback'
        actual, expected = getattr(nrpy_attr_wrapper, attr), callback
        assert actual is expected

    def test_doit_method_called_once(self, nrpy_attr_wrapper, callback):
        nrpy_attr_wrapper.doit()
        callback.assert_called_once()

    def test_doit_method_arguments(self, nrpy_attr_wrapper, callback):
        nrpy_attr_wrapper.doit()
        callback.assert_called_with(*self.TESTCASE_ARGS, **self.TESTCASE_KWARGS)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
