from enum import Enum

class DerivativeType(Enum):
    IRS = 'irs'
    CCS = 'ccs'
    FXFORWARD = 'fxforward'
    FXCALL = 'fxcall'
    FXPUT = 'fxput'
    
class Position(Enum):
    Active = 'active'
    Passive = 'passive'
    
class PaymentType(Enum):
    PD = 'physical delivery'
    CS = 'cash settle'