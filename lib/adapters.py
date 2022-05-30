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
