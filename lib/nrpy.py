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

class CoreParameters():
    """summary of step 2

    Attributes
    ----------
    coord_system: CoordSystem = CoordSystem.Spherical
    domain_size: int = 32
    sinh_width: float = 0.2
    sinh2_const_dr: float = 0.05
    symtp_bscale: float = 0.5
    rk_method: RkMethod = RkMethod.RK4
    fd_order: int = 4
    real: str = 'double'
    cfl_factor: float = 0.5
    lapse_condition = 'OnePlusLog'
    shift_condition = 'GammaDriving2ndOrder_Covariant'

    >>> core_parameters = CoreParameters()
    >>> print(core_parameters.coord_system)
    Spherical
    >>> isinstance(core_parameters.coord_system, CoordSystem)
    True
    >>> print(core_parameters.domain_size)
    32
    >>> print(core_parameters.sinh_width)
    0.2
    >>> print(core_parameters.sinhv2_const_dr)
    0.05
    >>> print(core_parameters.symtp_bscale)
    0.5
    >>> print(core_parameters.rk_method)
    RK4
    >>> isinstance(core_parameters.rk_method, RkMethod)
    True
    >>> print(core_parameters.fd_order)
    4
    >>> print(core_parameters.real)
    double
    >>> print(core_parameters.cfl_factor)
    0.5
    >>> print(core_parameters.lapse_condition)
    OnePlusLog
    >>> print(core_parameters.shift_condition)
    GammaDriving2ndOrder_Covariant
    """

    def __init__(self,**kwargs):
        self.coord_system = kwargs.get('coord_system', 'spherical')
        self.domain_size = kwargs.get('domain_size', 32)
        self.sinh_width = kwargs.get('sinh_width', 0.2)
        self.sinhv2_const_dr = kwargs.get('sinhv2_const_dr', 0.05)
        self.symtp_bscale = kwargs.get('symtp_bscale', 0.5)
        self.rk_method = kwargs.get('rk_method', 'rk4')
        self.fd_order = kwargs.get('fd_order', 4)
        self.real = kwargs.get('real', 'double')
        self.cfl_factor = kwargs.get('cfl_factor', 0.5)
        self.lapse_condition = kwargs.get('lapse_condition', 'OnePlusLog')
        self.shift_condition = kwargs.get('shift_condition', 'GammaDriving1ndOrder_Covariant')

    @property
    def coord_system(self):
        return self._coord_system

    @coord_system.setter
    def coord_system(self, value):
        self._coord_system = CoordSystem.pick(value)

    @property
    def rk_method(self):
        return self._rk_method

    @rk_method.setter
    def rk_method(self, value):
        self._rk_method = RkMethod.pick(value)
