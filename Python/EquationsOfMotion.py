from sympy import *
from Einstein import Spacetime
from Linearize import Linearize

class EOMs:
    def __init__(self, co, lagrangian, fields):
        self.lagrangian = lagrangian
        self.co = co
        self.dim = len(co)
        self.fields = fields
        self.num_fields = len(fields)
        self.dfields = [Function('dfield'+str(i))(*self.co) for i in range(self.num_fields)]
    

    def get_eoms(self):
        delta_lagrangian = Linearize(self.lagrangian, self.fields, self.dfields).linearize()
        eoms = [0 for i in range(self.num_fields)]
        for i in range(self.num_fields):
            e_l_eq = simplify(sum([diff(delta_lagrangian.diff(diff(self.dfields[i],self.co[j])),self.co[j]) for j in range(self.dim)])-delta_lagrangian.diff(self.dfields[i]))
            eoms[i] = e_l_eq
        return eoms
