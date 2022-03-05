'''unittest for supportive module'''
import unittest

from supportive import SupportiveClass, supportive_function


class TestSupportiveClass(unittest.TestCase):
    '''test supportive class functionality'''

    def test_say_method(self):
        '''test say method'''

        what_to_say = 'pen is mightier than the sword'

        support = SupportiveClass(what_to_say)
        what_was_said = support.say()

        expt = f'Supportive class says: {what_to_say}'
        self.assertEqual(what_was_said, expt, 'say method speaks the truth')


class TestSupportiveFunctions(unittest.TestCase):
    '''test supportive_function'''

    def test_support_message(self):
        '''test if support message is returned'''

        what_to_say = 'functions are mightier than classes'

        what_was_said = supportive_function(what_to_say)

        expt = f'Supportive function says: {what_to_say}'
        self.assertEqual(what_was_said, expt, 'supportive function supported')


if __name__ == '__main__':
    unittest.main()
