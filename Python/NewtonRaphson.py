from sympy import *
from PseudospectralGrid import Grid
from Linearize import Linearize
import numpy as np
import time

"""
This module defines an ODE object.

    parameters: 
        eqs:                ode's (sympy equation)
        bcs:                boundary conditions (sympy equation)
        fields:             fields (sympy function)
        dfields:            linearization of fields to find Jacobian (sympy function)
        params:             non-field parameters in ode's, like chemical potential (sympy symbol)
        grid:               the Chebyshev grid (sympy Matrix)
        co:                 the coordinate (sympy symbol)
        bounds:             the range of the coordinate (python tuple)

    methods:
        linearize_and_extract():                                takes the odes, linearizes, 
                                                                and extracts Jacobian
        build_matrices(current_fields, parameters, print time): takes the current values of fields
                                                                and parameters and builds the 
                                                                numerical Jacobian. Option to print the time 
                                                                it takes to build this matrix
        step(current_fields, parameters, print_time, as_numpy): finds delta field for a Newton Raphson step
                                                                option to use numpy arrays and linalg which
                                                                is faster but lower precision. if not, solves
                                                                using sympy matrix at high precision
"""

class ODE:
    def __init__(self, eqs, bcs, fields, dfields, params, grid, co, bounds):
        self.eqs = eqs
        self.bcs = bcs
        self.fields = fields
        self.dfields = dfields
        self.grid = grid
        self.NumPoints = grid.NumPoints
        self.D0 = grid.D0()
        self.D1 = grid.D1()
        self.D2 = grid.D2()
        self.r_vec = grid.get_grid()
        self.bounds = bounds
        self.num_fields = len(fields)
        self.co = co
        self.EQ_coef, self.BC_coef = self.linearize_and_extract()
        self.u_syms = fields
        self.up_syms = [field.diff(co) for field in fields]
        self.upp_syms = [field.diff(co, co) for field in fields]
        self.params = params

        # lambdify eqs and linearized eqs
        self.ode_arglist = (
            self.u_syms + self.up_syms + self.upp_syms + [self.co] + self.params
        )
        self.eqs_lambda = [
            lambdify(self.ode_arglist, eq, modules="sympy") for eq in self.eqs
        ]

        self.EQ_coef_lambda = [
            [
                [
                    lambdify(self.ode_arglist, self.EQ_coef[i][j][l], modules="sympy")
                    for l in range(3)
                ]
                for j in range(self.num_fields)
            ]
            for i in range(self.num_fields)
        ]

        # lambdify bcs
        self.bc_syms_left = []
        self.bc_d_syms_left = []
        self.bc_syms_right = []
        self.bc_d_syms_right = []

        for i in range(len(fields)):
            ll = symbols(f'ul{i}')
            dl = symbols(f'dul{i}')
            self.bc_syms_left.append(ll)
            self.bc_d_syms_left.append(dl)

            rr = symbols(f'ur{i}')
            dr = symbols(f'dur{i}')
            self.bc_syms_right.append(rr)
            self.bc_d_syms_right.append(dr)

            dk = [dl, dr]
            ck = [ll, rr]

            self.bcs = [
                [
                    self.bcs[k][l]
                    .subs(fields[i].diff(co).subs(co, bounds[l]), dk[l])
                    .subs(fields[i].subs(co, bounds[l]), ck[l])
                    for l in range(2)
                ]
                for k in range(self.num_fields)
            ]

            self.BC_coef = [
                [
                    [
                        [
                            self.BC_coef[m][k][j][l]
                            .subs(fields[i].diff(co).subs(co, bounds[k]), dk[k])
                            .subs(fields[i].subs(co, bounds[k]), ck[k])
                            for l in range(3)
                        ]
                        for j in range(self.num_fields)
                    ]
                    for k in range(2)
                ]
                for m in range(self.num_fields)
            ]

        bc_arglist_left = self.bc_syms_left + self.bc_d_syms_left + self.params
        bc_arglist_right = self.bc_syms_right + self.bc_d_syms_right + self.params
        bc_arglist = [bc_arglist_left, bc_arglist_right]

        self.bcs_lam = [
            [
                lambdify(bc_arglist[l], self.bcs[k][l], modules="sympy")
                for l in range(2)
            ]
            for k in range(self.num_fields)
        ]

        self.BC_coef_lam = [
            [
                [
                    [
                        lambdify(
                            bc_arglist[k],
                            self.BC_coef[i][k][j][l],
                            modules="sympy"
                        )
                        for l in range(3)
                    ]
                    for j in range(self.num_fields)
                ]
                for k in range(2)
            ]
            for i in range(self.num_fields)
        ]

    def linearize_and_extract(self):
        """Linearize equations of motion for Newton-Raphson method."""
        boundfields = [
            [fieldi.subs(self.co, self.bounds[i]) for fieldi in self.fields]
            for i in range(2)
        ]
        bounddfields = [
            [dfieldi.subs(self.co, self.bounds[i]) for dfieldi in self.dfields]
            for i in range(2)
        ]

        linEQ = [
            Linearize(eq, self.fields, self.dfields).linearize()
            for eq in self.eqs
        ]

        linBC = [
            [
                Linearize(self.bcs[i][j], self.fields, self.dfields).linearize() +
                Linearize(self.bcs[i][j], boundfields[j], bounddfields[j]).linearize()
                for j in range(2)
            ]
            for i in range(self.num_fields)
        ]

        EQ_coef = [
            [
                [
                    linEQ[i].diff(self.dfields[j].diff(self.co, self.co)),
                    linEQ[i].diff(self.dfields[j].diff(self.co)),
                    linEQ[i].diff(self.dfields[j])
                ]
                for j in range(self.num_fields)
            ]
            for i in range(self.num_fields)
        ]

        # Handle boundary derivatives using substitution
        fpp, fp = symbols('fpp fp')
        BC_coef = [
            [
                [
                    [
                        linBC[i][l]
                        .subs(self.dfields[j].diff(self.co, self.co)
                              .subs(self.co, self.bounds[l]), fpp).diff(fpp),
                        linBC[i][l]
                        .subs(self.dfields[j].diff(self.co)
                              .subs(self.co, self.bounds[l]), fp).diff(fp),
                        linBC[i][l].diff(self.dfields[j].subs(self.co, self.bounds[l]))
                    ]
                    for j in range(self.num_fields)
                ]
                for l in range(2)
            ]
            for i in range(self.num_fields)
        ]

        return EQ_coef, BC_coef

    def build_matrices(self, current_fields, parameters, print_time=False):
        """Build differentiation matrix and equations evaluated on current field configurations."""
        start = time.time()
        paramvals = [parameters[key] for key in self.params]

        eqs_cur = []
        d_vec = []

        for k in range(self.NumPoints + 1):
            uvals = [current_fields[i][k] for i in range(self.num_fields)]
            upvals = [(self.D1[k, :] * current_fields[i])[0, 0]
                      for i in range(self.num_fields)]
            uppvals = [(self.D2[k, :] * current_fields[i])[0, 0]
                       for i in range(self.num_fields)]

            if k == 0 or k == self.NumPoints:
                val_sub = uvals + upvals + paramvals
                idx = 0 if k == 0 else 1
                d_vec.append([
                    [
                        [
                            self.BC_coef_lam[i][idx][j][l](*val_sub)
                            for l in range(3)
                        ]
                        for j in range(self.num_fields)
                    ]
                    for i in range(self.num_fields)
                ])
                eqs_cur.append([
                    self.bcs_lam[i][idx](*val_sub)
                    for i in range(self.num_fields)
                ])
            else:
                val_sub = uvals + upvals + uppvals + [self.r_vec[k]] + paramvals
                eqs_cur.append([
                    self.eqs_lambda[i](*val_sub)
                    for i in range(self.num_fields)
                ])
                d_vec.append([
                    [
                        [
                            self.EQ_coef_lambda[i][j][l](*val_sub)
                            for l in range(3)
                        ]
                        for j in range(self.num_fields)
                    ]
                    for i in range(self.num_fields)
                ])

        end1 = time.time()
        if print_time:
            print(f"Time to build coeffs: {end1 - start:.3f} seconds")

        blocks = [
            [
                Matrix([
                    self.D0[k, :] * d_vec[k][i][j][2] +
                    self.D1[k, :] * d_vec[k][i][j][1] +
                    self.D2[k, :] * d_vec[k][i][j][0]
                    for k in range(self.NumPoints + 1)
                ])
                for j in range(self.num_fields)
            ]
            for i in range(self.num_fields)
        ]

        dmatrix = BlockMatrix(blocks).as_explicit()
        ematrix = BlockMatrix([
            [Matrix(eqs_cur)[:, i]] for i in range(self.num_fields)
        ]).as_explicit()

        end2 = time.time()
        if print_time:
            print(f"Time to build matrices: {end2 - start:.3f} seconds")

        return dmatrix, ematrix

    def step(self, current_fields, parameters, print_time=False, as_numpy=False):
        dmatrix, ematrix = self.build_matrices(current_fields, parameters, print_time)
        start = time.time()

        if as_numpy:
            dn = np.array(dmatrix).astype(np.float64)
            en = np.array(ematrix).astype(np.float64)
            df = Matrix(np.linalg.solve(dn, -en))
        else:
            df = dmatrix.solve(-ematrix)

        end = time.time()
        if print_time:
            print(f"Time to evaluate matrices: {end - start:.3f} seconds")

        return df

        
