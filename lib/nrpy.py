'''NRPY+ wrapper'''

import shutil
import os
import cmdline_helper as cmd

# This should be done by Makefile
class Ccode:
    """C code output directory builder; NOTE this should be handled by Makefile

    Attributes
    ----------
    root: str
        Root directory for generated C code and output
    ccodes: str
        C generated code name, child of root
    ouptut: str
        Output directory, child of ccodes

    Methods
    -------
    build() -> None
        Build directory structure; WARNING: Destructive

    Sample usage
    >>> ccode = Ccode()
    >>> print(ccode.root)
    bin
    >>> print(ccode.ccodes)
    bin/ccodes
    >>> print(ccode.output)
    bin/ccodes/output
    >>> ccode.output = 'output2' # change target output directory
    >>> print(ccode.output)
    bin/ccodes/output2
    """

    def __init__(self, root='bin', ccodes='ccodes', output='output'):
        self.root = root
        self.ccodes = ccodes
        self.output = output

    @property
    def root(self): # pylint: disable=missing-function-docstring
        return self._root

    @root.setter
    def root(self, root):
        self._root = os.path.join(root)

    def build_root(self):
        """build root directory"""
        shutil.rmtree(self.root, ignore_errors=True)
        cmd.mkdir(self.root)

    @property
    def ccodes(self):# pylint: disable=missing-function-docstring
        return self._ccodes

    @ccodes.setter
    def ccodes(self, ccodes):
        self._ccodes = os.path.join(self.root, ccodes)

    def build_ccodes(self):
        """build ccodes directory"""
        cmd.mkdir(self.ccodes)

    @property
    def output(self):# pylint: disable=missing-function-docstring
        return self._output

    @output.setter
    def output(self, output):
        self._output = os.path.join(self.ccodes, output)

    def build_output(self):
        """build output directory"""
        cmd.mkdir(self.output)

    def build(self):
        """Build directory structure defined by attribute names"""
        self.build_root()
        self.build_ccodes()
        self.build_output()

# class SupportiveClass: # pylint: disable=too-few-public-methods
#     '''Supportive class, can be very supportive
#     Sample usage:
#     >>> support = SupportiveClass('you can do it')
#     >>> support.say()
#     'Supportive class says: you can do it'
#     '''

#     def __init__(self, message):
#         self.message = message

#     def say(self):
#         '''share some supportive toughts'''
#         return f'Supportive class says: {self.message}'


# def supportive_function(message):
#     '''functions can also provide supportive messages'''
#     return f'Supportive function says: {message}'
