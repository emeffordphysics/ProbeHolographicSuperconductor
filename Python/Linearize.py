from sympy import *

""" This module linearizes sympy expressions via dE[field + eps*dfield]/deps """

class Linearize:
    def __init__(self, expr, fields, dfields):
        self.expr = expr
        self.fields = fields
        self.dfields = dfields
        self.num_fields = len(fields)

    def linearize(self):
        eps = symbols('eps')
        delta_fields = [
            (self.fields[i], self.fields[i] + eps*self.dfields[i]) for i in range(self.num_fields)]
        delta_expr = diff(self.expr.subs(delta_fields),eps).subs(eps,0)
        return delta_expr
