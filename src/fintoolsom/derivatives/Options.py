from datetime import date
from typing import Union
from collections.abc import Sequence
import numpy as np

from .Derivatives import Derivative
from .enums import PaymentType, Position

from ..rates import Rate
from .. import dates
        
#Yet to come