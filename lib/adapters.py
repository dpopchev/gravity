import os
from dataclasses import dataclass
from typing import Any
from enum import Enum, auto
import nrpy_local as nrpy

@dataclass
class CcodesDir:
    root: str = None
    outdir: str = None

    @classmethod
    def build(cls, root = "ccodesdir_default", outdir = 'output'):
        _root = root
        _outdir = os.path.join(root, outdir)
        self = cls(_root, _outdir)
        self.make_root()
        self.make_outdir()
        return self

    def clean(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def make_root(self):
        nrpy.cmd.mkdir(self.root)

    def make_outdir(self):
        nrpy.cmd.mkdir(self.outdir)

    def make_under_root(self, destination):
        _destination = os.path.join(self.root, destination)
        nrpy.cmd.mkdir(_destination)
        return _destination

@dataclass
class InterfaceParameter:
    parameter: str = None
    value: int = None
    representation: Any = None

    @classmethod
    def build(cls, parameter, value):
        self = cls(parameter, value)
        nrpy.par.set_parval_from_str(parameter, value)
        self.representation = nrpy.par.parval_from_str(self.parameter)
        return self

    def __str__(self):
        return self.representation

    def as_string(self):
        return self.__str__()

class CoordSystemVariant(Enum):
    SPHERICAL = auto()
    SINHSPHERICAL = auto()
    SINHSPHERICALV2 = auto()
    CYLINDRICAL = auto()
    SINHCYLINDRICAL = auto()
    SYMTP = auto()
    SINHSYMTP = auto()

    @classmethod
    def pick(cls, candidate):
        _candidate = candidate.strip()
        _candidate = _candidate.upper()

        members = cls.__members__.keys()
        matches = (m for m in members if _candidate == m)
        match = next(matches, None)

        if match is None:
            raise ValueError(f'Coordinate system {_candidate} candidate not found')

        return cls.__members__[match]

    @classmethod
    def supported(cls):
        nrpy_names = ['Spherical', 'SinhSpherical', 'SinhSphericalv2', 'Cylindrical', 'SinhCylindrical', 'SymTP', 'SinhSymTP']
        members = cls.__members__.keys()
        return { member: nrpy_name for member, nrpy_name in zip(members, nrpy_names) }

    def __str__(self):
        supported = self.__class__.supported()
        return supported[self.name]

@dataclass
class CoordSystem:
    name: CoordSystemVariant = None
    domain_size: int = None
    sinh_width: float = None
    sinhv2_const_dr: float = None
    symtp_bscale: float = None

    @classmethod
    def build_spherical(cls, domain_size=32):
        name = CoordSystemVariant.SPHERICAL
        return cls(name=name, domain_size=domain_size)

    @classmethod
    def build_sinh_spherical(cls, domain_size=32, sinh_width=0.2):
        name = CoordSystemVariant.SINHSPHERICAL
        return cls(name=name, domain_size=domain_size, sinh_width=sinh_width)

    @classmethod
    def build_sinh_spherical_v2(cls, domain_size=32, sinhv2_const_dr=0.05):
        name = CoordSystemVariant.SINHSPHERICALV2
        return cls(name=name, domain_size=domain_size, sinhv2_const_dr=sinhv2_const_dr)

    @classmethod
    def build_sinh_cylindrical(cls, domain_size=32, sinh_width=0.2):
        name = CoordSystemVariant.SINHCYLINDRICAL
        return cls(name=name, domain_size=domain_size, sinh_width=sinh_width)

    @classmethod
    def build_sinh_symtp(cls, domain_size=32, sinh_width=0.2):
        name = CoordSystemVariant.SINHSYMTP
        return cls(name=name, domain_size=domain_size, sinh_width=sinh_width)

    @classmethod
    def build_symtp_bscale(cls, domain_size=32, symtp_bscale=0.5):
        name = CoordSystemVariant.SINHSYMTP
        return cls(name=name, domain_size=domain_size, symtp_bscale=symtp_bscale)

class IntegratorVariant(Enum):
    EULER = auto()
    RK2HEUN = auto()
    RK2MP = auto()
    RK2RALSTON = auto()
    RK3 = auto()
    RK3HEUN = auto()
    RK3RALSTON = auto()
    SSPRK3 = auto()
    RK4 = auto()
    DP5 = auto()
    DP5ALT = auto()
    CK5 = auto()
    DP6 = auto()
    L6 = auto()
    DP8 = auto()

    @classmethod
    def pick(cls, candidate):
        _candidate = candidate.strip()
        _candidate = _candidate.upper()
        _candidate = _candidate.replace(' ', '')

        members = cls.__members__.keys()
        matches = (m for m in members if _candidate == m)
        match = next(matches, None)

        if match is None:
            raise ValueError(f'Coordinate system {_candidate} candidate not found')

        return cls.__members__[match]

    @classmethod
    def supported(cls):
        nrpy_names = [ "Euler", "RK2 Heun", "RK2 MP", "RK2 Ralston", "RK3", "RK3 Heun", "RK3 Ralston", "SSPRK3", "RK4", "DP5", "DP5alt", "CK5", "DP6", "L6", "DP8" ]
        members = cls.__members__.keys()
        return { member: nrpy_name for member, nrpy_name in zip(members, nrpy_names) }

    def as_string(self):
        return self.__str__()

    def __str__(self):
        supported = self.__class__.supported()
        return supported[self.name]

@dataclass
class NumericalIntegration:
    name: IntegratorVariant = None
    fd_order: int = None
    real: str = None
    cfl_factor: float = None
    lapse_condition: str = None
    shift_condition: str = None

    @classmethod
    def build(cls, name=IntegratorVariant.RK4, fd_order=4, real="double", cfl_factor=0.5, lapse_condition='OnePlusLog', shift_condition="GammaDriving2ndOrder_Covariant"):
        return cls(name, fd_order, real, cfl_factor, lapse_condition, shift_condition)

    @property
    def rk_order(self):
        _, entry = nrpy.Butcher_dict[self.name.as_string()]
        return entry

@dataclass
class CcodePrototypeArgument:
    name: str = None
    address_order: int = None

    def as_string(self):
        return self.__str__()

    def __str__(self):
        return f'{"&"*self.address_order}{self.name}'

@dataclass
class CcodePrototype:
    name: str = None
    arguments: Any = None

    def as_string(self):
        return self.__str__()

    def __str__(self):
        arguments = ', '.join((str(a) for a in self.arguments))
        return f'{self.name}({arguments});'

@dataclass
class NrpyAttrWrapper:
    name: str = None
    callback: Any = None
    args: Any = None
    kwargs: Any = None

    def doit(self):
        result = self.callback(*self.args, **self.kwargs)
        return result
