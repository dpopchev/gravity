import pytest

from adapters import CcodePrototype
from collections import namedtuple

CcodePrototypeStub = namedtuple('CcodePrototype', 'name arguments as_string')

class TestZeroAddressArgument:
    TESTCASE_ARGUMENTS = ('&rfmstruct', '&params', 'RK_INPUT_GFS', 'auxevol_gfs')
    TESTCASE_EXPECTED = 'Ricci_eval(&rfmstruct, &params, RK_INPUT_GFS, auxevol_gfs); // DPythonMark'
    TESTCASE = CcodePrototypeStub('Ricci_eval', TESTCASE_ARGUMENTS, TESTCASE_EXPECTED)

    @pytest.fixture
    def ccode_prototype(self):
        _ccode_prototype = CcodePrototype(self.TESTCASE.name, self.TESTCASE.arguments)
        return _ccode_prototype

    def test_name_attr(self, ccode_prototype):
        attr = 'name'
        actual, expected = map(lambda _: getattr(_, attr), (ccode_prototype, self.TESTCASE))
        assert actual == expected

    def test_as_string(self, ccode_prototype):
        attr = 'as_string'
        actual, expected = map(lambda _: getattr(_, attr), (ccode_prototype, self.TESTCASE))
        actual = actual()
        assert actual == expected

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
