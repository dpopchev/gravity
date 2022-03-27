'''unittest for main script function'''
import unittest
from unittest.mock import patch
import io

import importlib.machinery
import importlib.util
from pathlib import Path


@unittest.skip('template')
class TestMainFunctionStdoutPrint(unittest.TestCase):
    '''main function stdout verification'''

    def setUp(self):
        cwd = Path(__file__).parent
        mainexec_path = str(cwd.joinpath('..', '..', 'src', 'mainexec'))

        loader = importlib.machinery.SourceFileLoader(
            'mainexec', mainexec_path)
        spec = importlib.util.spec_from_loader('mainexec', loader)
        mainexec = importlib.util.module_from_spec(spec)
        loader.exec_module(mainexec)

        self.mainexec = mainexec

    def test_main_function_stdout(self):
        '''basic std out inf printed by main function'''

        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:

            self.mainexec.main()
            got = fake_stdout.getvalue()

            expt = (
                "Hello from mainexec, a stand alone python script\n"
                "It is taking advantage of supportive module, found under lib/\n"
                "Supportive class says: here is my number\n"
                "Supportive function says: so call me maybe\n")

            self.assertMultiLineEqual(
                got, expt, 'Main function prints info to stdout')


if __name__ == '__main__':
    unittest.main()
