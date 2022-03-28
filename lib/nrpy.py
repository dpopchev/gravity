'''NRPY+ wrapper'''

import shutil
import os
import cmdline_helper as cmd

from enum import Enum

class CoordSystem(Enum):
    SPHERICAL = 'Spherical'
    SINHSPHERICAL = 'SinhSpherical'
    SINHSPHERICALV2 = 'SinhSphericalv2'
    CYLINDRICAL = 'Cylindrical'
    SINHCYLINDRICAL = 'SinhCylindrical'
    SYMTP = 'SymTP'
    SINHSYMTP = 'SinhSymTP'

    @classmethod
    def pick(cls, system):
        normalized = system.upper()
        matches = (m for m in cls.__members__.items() if m[0] == normalized)

        match = next(matches, None)
        if match is None:
            raise AttributeError('No matches')

        match0 = next(matches, None)
        if match0 is not None:
            raise AttributeError('Too many matches')

        return match[1]

    def __str__(self):
        return f'{self.value}'

    def as_string(self):
        return self.__str__()

class RkMethod(Enum):
    EULER = 'Euler'
    RK2HEUN = 'RK2 Heun'
    RK2MP = 'RK2 MP'
    RK2RALSTON = 'RK2 Ralston'
    RK3 = 'RK3'
    RK3HEUN = 'RK3 Heun'
    RK3RALSTON = 'RK3 Ralston'
    SSPRK3 = 'SSPRK3'
    RK4 = 'RK4'
    DP5 = 'DP5'
    DP5ALT = 'DP5alt'
    CK5= 'CK5'
    DP6 = 'DP6'
    L6 = 'L6'
    DP8 = 'DP8'

    @classmethod
    def pick(cls, system):
        normalized = system.upper()
        matches = (m for m in cls.__members__.items() if m[0] == normalized)

        match = next(matches, None)
        if match is None:
            raise AttributeError('No matches')

        match0 = next(matches, None)
        if match0 is not None:
            raise AttributeError('Too many matches')

        return match[1]

    def __str__(self):
        return f'{self.value}'

    def as_string(self):
        return self.__str__()

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
