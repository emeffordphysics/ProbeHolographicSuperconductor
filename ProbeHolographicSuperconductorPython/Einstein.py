from sympy import *

"""
This defines a Spacetime class in terms of the line element and coordinates.
The conventions follow Sean Carrol "Spacetime and Geometry"

Inputs: 
    ds2:    line element, a sympy symbol
    co:     coordinates, sympy symbols
    dco:    differentials of coordinates, sympy symbols

Outputs:
    metric():           returns the metric as a sympy matrix. index order matches co
    inverse_metric():   returns the inverse metric as a sympy matrix. index order matches co
    christoffel():      returns the christoffel symbols as a python list of lists. 
                        index order for Gamma^{mu}_{sigma nu} is [mu][sigma][nu]
    riemann():          returns the riemann tensor as a python list of lists.
                        index order for R^{mu}_{nu rho sigma} is [mu][nu][rho][sigma]
    ricci_tensor():     returns the ricci tensor as python list of lists.
                        index order for R^{mu}_{  nu} is [mu][nu]
    ricci_scalar():     returns the ricci scalar as a scalar
"""
class Spacetime:
    def __init__(self, ds2, co, dco):
        self.ds2 = ds2
        self.co = co
        self.dco = dco
        self.nc = len(co)

    def metric(self):
        return Matrix([[Rational(1,2)*diff(self.ds2,self.dco[i],self.dco[j]) if i==j else diff(self.ds2,self.dco[i],self.dco[j]) for j in range(self.nc)] for i in range(self.nc)])

    def inverse_metric(self):
        return self.metric().inv()
    
    def det_met(self):
        return self.metric().det()
    
    def christoffel(self):
        g = self.metric()
        ginv = self.inverse_metric()
        return [[[simplify(sum([Rational(1,2)*ginv[i,l]*(diff(g[k,l],self.co[j])+diff(g[j,l],self.co[k])-diff(g[j,k],self.co[l])) for l in range(self.nc)])) for k in range(self.nc)] for j in range(self.nc)] for i in range(self.nc)]
    
    def riemann(self):
        cs = self.christoffel()
        return [[[[simplify(diff(cs[i][l][j],self.co[k])-diff(cs[i][k][j],self.co[l])+sum([cs[i][k][m]*cs[m][l][j]-cs[i][l][m]*cs[m][k][j] for m in range(self.nc)])) for l in range(self.nc)] for k in range(self.nc)] for j in range(self.nc)] for i in range(self.nc)]

    def ricci_tensor(self):
        re_t = self.riemann()
        return [[simplify(sum([re_t[i][j][i][k] for i in range(self.nc)])) for k in range(self.nc)] for j in range(self.nc)]

    def ricci_scalar(self):
        ginv = self.inverse_metric()
        ri_t = self.ricci_tensor()
        return simplify(sum([sum([ginv[i,j]*ri_t[i][j] for i in range(self.nc)]) for j in range(self.nc)]))




