import pytest

from adapters import CcodePrototypeArgument
from collections import namedtuple

CcodePrototypeArgumentStub = namedtuple('CcodePrototypeArgumentStub', 'name address_order as_string')

class TestZeroAddressArgument:
    TESTCASE = CcodePrototypeArgumentStub('params', 0, 'params')

    @pytest.fixture
    def ccode_prototype_argument(self):
        _ccode_prototype_argument = CcodePrototypeArgument(self.TESTCASE.name,
                                                           self.TESTCASE.address_order)
        return _ccode_prototype_argument

    def test_name_attr(self, ccode_prototype_argument):
        attr = 'name'
        actual, expected = map(lambda _: getattr(_, attr), (ccode_prototype_argument, self.TESTCASE))
        assert actual == expected

    def test_address_order_attr(self, ccode_prototype_argument):
        attr = 'address_order'
        actual, expected = map(lambda _: getattr(_, attr), (ccode_prototype_argument, self.TESTCASE))
        assert actual == expected

    def test_str_method(self, ccode_prototype_argument):
        attr = 'as_string'
        actual, expected = map(lambda _: getattr(_, attr), (ccode_prototype_argument, self.TESTCASE))
        actual = actual()
        assert actual == expected

class TestOneAddressArgument:
    TESTCASE = CcodePrototypeArgumentStub('params', 1, '&params')

    @pytest.fixture
    def ccode_prototype_argument(self):
        _ccode_prototype_argument = CcodePrototypeArgument(self.TESTCASE.name,
                                                           self.TESTCASE.address_order)
        return _ccode_prototype_argument

    def test_name_attr(self, ccode_prototype_argument):
        attr = 'name'
        actual, expected = map(lambda _: getattr(_, attr), (ccode_prototype_argument, self.TESTCASE))
        assert actual == expected

    def test_address_order_attr(self, ccode_prototype_argument):
        attr = 'address_order'
        actual, expected = map(lambda _: getattr(_, attr), (ccode_prototype_argument, self.TESTCASE))
        assert actual == expected

    def test_str_method(self, ccode_prototype_argument):
        attr = 'as_string'
        actual, expected = map(lambda _: getattr(_, attr), (ccode_prototype_argument, self.TESTCASE))
        actual = actual()
        assert actual == expected

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
