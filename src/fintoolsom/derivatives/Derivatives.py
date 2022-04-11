import numpy as np

class Derivative:
    def __init__(self, operation_number, derivative_type, payment_type, collateral_index_name=None, compensation_currency=None):
        self.operation_number = operation_number
        self.derivative_type = derivative_type
        self.collateral_index_name = collateral_index_name
        self.payment_type = payment_type
        self.compensation_currency = compensation_currency