from datetime import date
from typing import Union
from collections.abc import Sequence
import numpy as np

from .Derivatives import Derivative
from .enums import PaymentType, Position

from ..rates import Rate
from .. import dates

class ForwardLeg:
    def __init__(self) -> None:
        pass
        
class FXForward(Derivative):
    def __init__(self, **kwargs):
        super.__init__(**kwargs)
        self.legs = {}
        self.legs[Position.Active] = kwargs['active_leg']
        self.legs[Position.Passive] = kwargs['passive_leg']