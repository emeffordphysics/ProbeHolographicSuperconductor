import time

import numpy as np
import scipy as scp
from sympy import (Matrix, BlockMatrix, Rational, eye, symbols, lambdify)
from PseudospectralGrid import Grid
from Linearize import Linearize
import EigenvalueRefinement

"""
This module defines an Eigenvalue object.

    parameters: 
        eqs:                linearized ode's (sympy equation)
        bcs:                boundary conditions (sympy equation)
        fields:             linearized fields (sympy function)
        dfields:            linearization of linearized fields to find Jacobian (sympy function)
        backgroundfields:   background fields (sympy function)
        params:             non-field parameters in ode's, like wavevector (sympy symbol)
        omega:              variable that serves as eigenvalue (sympy symbol)
        grid:               the Chebyshev grid (sympy Matrix)
        co:                 the coordinate (sympy symbol)
        bounds:             the range of the coordinate (python list)

    methods:
        linearize_and_extract():                                            takes the odes, linearizes, 
                                                                            and extracts Jacobian
        build_matrices(current_fields, parameters, print time):             takes the current values of background 
                                                                            fields and parameters and builds the 
                                                                            numerical Jacobian. Option to print the time 
                                                                            it takes to build this matrix
        find_eigenvalue(current_fields, parameters, print_time, as_numpy):  solves the generalized eigenvalue problem
            
                                                                            A v= omega B v

                                                                            for omega. Takes current values of fields and 
                                                                            parameters, passes these to build_matrices, then 
                                                                            builds the matrices A and B. This is a quadratic 
                                                                            eigenvalue problem, so v = (omega v_0, v_0) where 
                                                                            v_0 is the eigenvector and the matrix
                                                                            A = [[E_1, E_0],[I, 0]] and B = [[-E_2, 0],[0,I]] 
                                                                            where E_i is the coefficient of omega^i in the ode
"""


class Eigenvalue:
    def __init__(
        self, eqs, bcs, fields, dfields, backgroundfields,
        params, omega, grid, co, bounds, precision
    ):
        self.eqs = eqs
        self.bcs = bcs
        self.fields = fields
        self.dfields = dfields
        self.backgroundfields = backgroundfields
        self.omega = omega
        self.grid = grid
        self.NumPoints = grid.NumPoints
        self.D0 = grid.D0()
        self.D1 = grid.D1()
        self.D2 = grid.D2()
        self.r_vec = grid.get_grid()
        self.bounds = bounds
        self.num_fields = len(fields)
        self.num_back = len(backgroundfields)
        self.co = co
        self.precision = precision

        self.EQ_coef, self.BC_coef = self.linearize_and_extract()

        self.u_syms = backgroundfields
        self.up_syms = [field.diff(co) for field in backgroundfields]
        self.upp_syms = [field.diff(co, co) for field in backgroundfields]
        self.params = params

        # Lambdify eqs and linearized eqs
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

        # Lambdify boundary conditions
        self.bc_syms_left = []
        self.bc_d_syms_left = []
        self.bc_syms_right = []
        self.bc_d_syms_right = []
        for i in range(self.num_back):
            ll = symbols(f'ul{i}')
            dl = symbols(f'dul{i}')
            rr = symbols(f'ur{i}')
            dr = symbols(f'dur{i}')

            self.bc_syms_left.append(ll)
            self.bc_d_syms_left.append(dl)
            self.bc_syms_right.append(rr)
            self.bc_d_syms_right.append(dr)

            dk = [dl, dr]
            ck = [ll, rr]

            self.bcs = [
                [
                    self.bcs[k][l]
                    .subs(backgroundfields[i].diff(co).subs(co, bounds[l]), dk[l])
                    .subs(backgroundfields[i].subs(co, bounds[l]), ck[l])
                    for l in range(2)
                ]
                for k in range(self.num_fields)
            ]

            self.BC_coef = [
                [
                    [
                        [
                            self.BC_coef[m][k][j][l]
                            .subs(
                                backgroundfields[i].diff(co).subs(co, bounds[k]),
                                dk[k]
                            )
                            .subs(backgroundfields[i].subs(co, bounds[k]), ck[k])
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
                            bc_arglist[k], self.BC_coef[i][k][j][l], modules="sympy"
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
        # Linearize equations for Newton-Raphson method
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
                Linearize(self.bcs[i][j], self.fields, self.dfields).linearize()
                + Linearize(self.bcs[i][j], boundfields[j], bounddfields[j]).linearize()
                for j in range(2)
            ]
            for i in range(self.num_fields)
        ]

        # Extract derivative coefficients: Af'' + Bf' + Cf
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

        # Sub dummy variables for BC derivatives
        fpp, fp = symbols('fpp fp')
        BC_coef = [
            [
                [
                    [
                        linBC[i][l]
                        .subs(self.dfields[j].diff(self.co, self.co).subs(self.co, self.bounds[l]), fpp)
                        .diff(fpp),
                        linBC[i][l]
                        .subs(self.dfields[j].diff(self.co).subs(self.co, self.bounds[l]), fp)
                        .diff(fp),
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
        start = time.time()
        paramvals = [parameters[key] for key in self.params]

        d_vec = []
        for k in range(self.NumPoints + 1):
            uvals = [current_fields[i][k] for i in range(self.num_back)]
            upvals = [(self.D1[k, :] * current_fields[i])[0, 0]
                      for i in range(self.num_back)]
            uppvals = [(self.D2[k, :] * current_fields[i])[0, 0]
                       for i in range(self.num_back)]

            if k == 0 or k == self.NumPoints:
                val_sub = uvals + upvals + paramvals
                side = 0 if k == 0 else 1
                d_vec.append([
                    [
                        [
                            self.BC_coef_lam[i][side][j][l](*val_sub)
                            for l in range(3)
                        ]
                        for j in range(self.num_fields)
                    ]
                    for i in range(self.num_fields)
                ])
            else:
                val_sub = uvals + upvals + uppvals + [self.r_vec[k]] + paramvals
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

        if print_time:
            print(f"Time to build coeffs: {time.time() - start:.3f}s")

        blocks = [
            [
                Matrix([
                    self.D0[kk, :] * d_vec[kk][i][j][2]
                    + self.D1[kk, :] * d_vec[kk][i][j][1]
                    + self.D2[kk, :] * d_vec[kk][i][j][0]
                    for kk in range(self.NumPoints + 1)
                ])
                for j in range(self.num_fields)
            ]
            for i in range(self.num_fields)
        ]

        dmatrix = BlockMatrix(blocks).as_explicit()

        if print_time:
            print(f"Time to build matrices: {time.time() - start:.3f}s")

        return dmatrix

    def find_eigenvalue(self, current_fields, parameters,
                        print_time=False, as_numpy=False, desired_accuracy = 1e-12, max_norm=1, precision = 50):
        dmatrix = self.build_matrices(current_fields, parameters, print_time)
        dmat_len = len(dmatrix[:, 0])

        domega_0 = dmatrix.subs(self.omega, 0)
        domega_1 = dmatrix.diff(self.omega).subs(self.omega, 0)
        domega_2 = Rational(1, 2) * dmatrix.diff(self.omega, self.omega).subs(self.omega, 0)

        zero_mat = 0 * domega_0
        id_mat = eye(dmat_len)

        Amat = BlockMatrix([
            [domega_1, domega_0],
            [-id_mat, zero_mat]
        ]).as_explicit()

        Bmat = BlockMatrix([
            [domega_2, zero_mat],
            [zero_mat, id_mat]
        ]).as_explicit()


        # print(Amat)
        # print(Bmat)

        if as_numpy:
            An = np.array(Amat).astype(np.clongdouble)
            Bn = np.array(Bmat).astype(np.clongdouble)
            eigs = scp.linalg.eigvals(An, -Bn)
        else:
            eigs, vecs = EigenvalueRefinement.generalized_eigen_refine(Amat,-Bmat, max_norm = max_norm, prec=precision)
            # residuals = []
            # for lam, v in zip(eigs, vecs):
            #     residuals+=[EigenvalueRefinement.compute_residual(
            #         EigenvalueRefinement.sympy_to_mpmath_matrix(-sympyA), 
            #         EigenvalueRefinement.sympy_to_mpmath_matrix(-sympyB), lam, v)]


        if print_time:
            print(f"Time to evaluate eigenvalues: {time.time():.3f}s")

        return eigs

