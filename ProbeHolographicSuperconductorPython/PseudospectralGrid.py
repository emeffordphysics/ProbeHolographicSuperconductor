from sympy import *

class Grid:
    def __init__(self, NumPoints, l_min, l_max, precision = 15):
        self.NumPoints = NumPoints
        self.l_min = l_min
        self.l_max = l_max
        self.precision = precision

    def get_grid(self):
        return Matrix([N((self.l_max+self.l_min)/2 - (self.l_max-self.l_min)/2*cos(pi*i/self.NumPoints), self.precision) for i in range(self.NumPoints+1)])
    
    def D0(self):
        return eye(self.NumPoints+1)
    
    def D1(self):
        l_data = self.get_grid()
        a_l = [prod([1 if j==k else l_data[j]-l_data[k] for k in range(self.NumPoints+1)]) for j in range(self.NumPoints+1)]
        D1_l = [[sum([0 if k==j else 1/(l_data[j]-l_data[k]) for k in range(self.NumPoints+1)]) if i==j else a_l[i]/a_l[j]/(l_data[i]-l_data[j]) for j in range(self.NumPoints+1)] for i in range(self.NumPoints+1)]
        return Matrix(D1_l)
    
    def D2(self):
        D1_l = self.D1()
        return D1_l*D1_l