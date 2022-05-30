import os
from dataclasses import dataclass
import nrpy_local as nrpy
from typing import Any

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
