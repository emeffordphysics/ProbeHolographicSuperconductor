# from sympy import Matrix, nsimplify
# from mpmath import mp, mpf, mpc, matrix, chop, almosteq
# import numpy as np

# mp.dps = 50  # Set decimal precision

# def sympy_to_mpmath(M):
#     """Convert sympy Matrix to mpmath matrix with mpc entries."""
#     rows, cols = M.shape
#     M_mpm = matrix(rows, cols)
#     for i in range(rows):
#         for j in range(cols):
#             val = M[i, j]
#             # Convert sympy complex numbers to mpmath mpc
#             if val.is_complex:
#                 real_part = mpf(val.as_real_imag()[0])
#                 imag_part = mpf(val.as_real_imag()[1])
#                 M_mpm[i, j] = mpc(real_part, imag_part)
#             else:
#                 M_mpm[i, j] = mpf(val)
#     return M_mpm

# def refine_eigenpair(A, B, v_init, tol=1e-30, max_iter=50):
#     """Refine a single eigenpair (lambda, v) using Rayleigh quotient iteration."""
#     A_mpm = sympy_to_mpmath(A)
#     B_mpm = sympy_to_mpmath(B)
#     v = matrix([mpc(x) for x in v_init])
#     v = v / mp.norm(v)

#     lam = mpf(0)
#     for _ in range(max_iter):
#         Av = A_mpm * v
#         Bv = B_mpm * v
#         numerator = (v.H * Av)[0]
#         denominator = (v.H * Bv)[0]

#         # Avoid division by near-zero denominator
#         if abs(denominator) < mpf("1e-50"):
#             lam_new = mpf(0)
#         else:
#             lam_new = numerator / denominator

#         # Solve (A - lam_new * B) w = Bv for w
#         M = A_mpm - lam_new * B_mpm
#         try:
#             w = mp.lu_solve(M, Bv)
#         except ZeroDivisionError:
#             # Singular matrix; fallback: skip refinement
#             break

#         w_norm = mp.norm(w)
#         if w_norm == 0:
#             break
#         v_new = w / w_norm

#         # Check convergence on eigenvector and eigenvalue
#         if mp.norm(v - v_new) < tol and abs(lam_new - lam) < tol:
#             lam = lam_new
#             v = v_new
#             break
#         lam = lam_new
#         v = v_new

#     return lam, v

# def generalized_eigen_solve(A, B, num_eigs=None):
#     """Solve generalized eigenproblem A x = lambda B x with refinement and deflation."""
#     import scipy.linalg

#     # Convert sympy matrices to numpy arrays of floats for initial eigensolve (low precision)
#     A_np = np.array(A.tolist()).astype(np.complex128)
#     B_np = np.array(B.tolist()).astype(np.complex128)

#     # Use scipy.linalg.eig to get initial approximations (complex eigenvalues and eigenvectors)
#     eigvals, eigvecs = scipy.linalg.eig(A_np, B_np)

#     if num_eigs is None:
#         num_eigs = len(eigvals)

#     refined_eigs = []
#     refined_vecs = []

#     used_indices = set()
#     for i in range(len(eigvals)):
#         if len(refined_eigs) >= num_eigs:
#             break
#         if i in used_indices:
#             continue

#         v_init = eigvecs[:, i]
#         # Convert initial vector to mpmath matrix column vector
#         v_init_mpm = matrix([[mpc(x.real, x.imag)] for x in v_init])

#         # Refine this eigenpair
#         lam, v = refine_eigenpair(A, B, v_init_mpm)

#         # Deflation: Remove converged eigenvector component from subsequent vectors
#         refined_eigs.append(lam)
#         refined_vecs.append(v)

#         # Mark close eigenvalues as used to avoid duplicates
#         for j, ev in enumerate(eigvals):
#             if j not in used_indices:
#                 if abs(ev - lam) < 1e-12:
#                     used_indices.add(j)

#     return refined_eigs, refined_vecs

# # Example usage with your matrices A_sympy and B_sympy
# # A_sympy and B_sympy must be sympy.Matrix instances

# if __name__ == "__main__":
#     # Example small test matrices (replace with your matrices)
#     A_sympy = Matrix([[1, 2], [3, 4]])
#     B_sympy = Matrix([[2, 0], [0, 2]])

#     eigvals, eigvecs = generalized_eigen_solve(A_sympy, B_sympy)

#     for i, (lam, v) in enumerate(zip(eigvals, eigvecs)):
#         print(f"Eigenvalue {i}: {lam}")
#         print(f"Eigenvector {i}: {v}")

import sympy as sp
import numpy as np
import scipy.linalg
import mpmath
import gmpy2

# Set mpmath precision (decimal digits)
mpmath.mp.dps = 50

def sympy_to_mpmath_matrix(M):
    """
    Convert SymPy Matrix to mpmath matrix with mpmath.mpc entries.
    Handles complex and real entries robustly.
    """
    rows = M.rows
    cols = M.cols
    mat = mpmath.zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            entry = M[i, j]
            try:
                val = entry.evalf()
            except AttributeError:
                val = entry
            z = complex(val)
            mat[i, j] = mpmath.mpc(float(z.real), float(z.imag))
    return mat

def convert_vector_to_mpmath(v_in):
    """
    Convert iterable of complex-like entries to mpmath column vector of mpmath.mpc.
    """
    v_out = mpmath.zeros(len(v_in), 1)
    for i in range(len(v_in)):
        z = complex(v_in[i])
        v_out[i] = mpmath.mpc(float(z.real), float(z.imag))
    return v_out


def refine_single_eigenpair_mpmath(A_mp, B_mp, lam0, v0, tol=1e-30, maxiter=30, switch_iter=5, high_prec=50):
    """
    Refine one eigenpair (lam0, v0) for generalized eigenproblem A v = lam B v
    using Newton iteration in mpmath with adaptive precision and LU caching.
    """
    low_prec = 30
    mpmath.mp.dps = low_prec

    lam = mpmath.mpc(complex(lam0))
    v = convert_vector_to_mpmath(v0)
    v /= mpmath.norm(v)

    n = A_mp.rows
    lu_cache = {}
    piv_cache = {}

    def solve_shifted_system(vec, shift):
        key = (float(shift.real), float(shift.imag))
        if key not in lu_cache:
            M = A_mp - shift * B_mp
            lu, piv = mpmath.lu(M)
            lu_cache[key] = lu
            piv_cache[key] = piv
        return mpmath.lu_solve(lu_cache[key], piv_cache[key], vec)

    for k in range(maxiter):
        Av = A_mp * v
        Bv = B_mp * v
        r = Av - lam * Bv
        res_norm = mpmath.norm(r)
        if res_norm < tol:
            break

        vH = v.H
        num = vH * Av
        den = vH * Bv
        lam = num[0,0] / den[0,0]

        try:
            delta = solve_shifted_system(r, lam)
        except Exception:
            break

        v = v - delta

        if (k % 5) == 0:
            v /= mpmath.norm(v)

        if k == switch_iter:
            mpmath.mp.dps = high_prec
            lam = mpmath.mpc(complex(lam))
            v = convert_vector_to_mpmath(v)

    v /= mpmath.norm(v)
    return lam, v


def generalized_eigen_refine(sympyA, sympyB, refine_tol=1e-30, max_refine_iter=30, max_norm=10, prec=50):
    """
    Solve generalized eigenproblem sympyA v = lambda sympyB v,
    refine eigenpairs with Newton iteration only if |lambda| < max_norm,
    at specified decimal precision 'prec'.
    """
    A_np = np.array(sympyA.evalf(15).tolist(), dtype=np.complex128)
    B_np = np.array(sympyB.evalf(15).tolist(), dtype=np.complex128)

    eigvals, eigvecs = scipy.linalg.eig(A_np, B_np)

    finite_idx = [i for i, val in enumerate(eigvals)
                  if np.isfinite(val) and abs(val) < max_norm]

    A_mp = sympy_to_mpmath_matrix(sympyA)
    B_mp = sympy_to_mpmath_matrix(sympyB)

    refined_lams = []
    refined_vecs = []

    for i in finite_idx:
        lam0 = eigvals[i]
        v0 = eigvecs[:, i]

        lam_r, v_r = refine_single_eigenpair_mpmath(
            A_mp, B_mp, lam0, v0,
            tol=refine_tol, maxiter=max_refine_iter, high_prec=prec)

        refined_lams.append(lam_r)
        refined_vecs.append(v_r)

    return refined_lams, refined_vecs

def compute_residual(A, B, lam, v):
    """
    Compute relative residual: ||A v - λ B v|| / (||A v|| + |λ| * ||B v||)
    using mpmath matrices.
    """
    Av = A * v
    Bv = B * v
    num = mpmath.norm(Av - lam * Bv)
    denom = mpmath.norm(Av) + abs(lam) * mpmath.norm(Bv)
    return float(num / denom) if denom != 0 else float(num)


if __name__ == "__main__":
    # Example matrices
    N = 60
    rng = np.random.default_rng(42)

    A_np = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    B_np = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))

    sympyA = sp.Matrix(A_np.tolist())
    sympyB = sp.Matrix(B_np.tolist())

    refined_eigs, refined_vecs = generalized_eigen_refine(sympyA, sympyB, max_norm=1/2, prec = 100)

    print("Refined eigenvalues and residuals:")
    for lam, v in zip(refined_eigs, refined_vecs):
        print(f"  λ = {complex(float(lam.real), float(lam.imag))}")
        res = compute_residual(sympy_to_mpmath_matrix(sympyA), sympy_to_mpmath_matrix(sympyB), lam, v)
        print(f"  Residual: {res:.2e}")

    print("\n--- Tests with badly conditioned matrices ---")
    print("\n--- TEST 1: Singular with finite eigenvalues ---")
    A_data1 = [
        [1, 2, 0, 0],
        [3, 4, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    B_data1 = [
        [5, 6, 0, 0],
        [7, 8, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    sympyA1 = sp.Matrix(A_data1)
    sympyB1 = sp.Matrix(B_data1)

    eigvals1, eigvecs1 = generalized_eigen_refine(sympyA1, sympyB1, max_norm=100, prec = 50)

    print("Refined eigenvalues and residuals:")
    for lam, v in zip(eigvals1, eigvecs1):
        print(f"  λ = {complex(float(lam.real), float(lam.imag))}")
        res = compute_residual(sympy_to_mpmath_matrix(sympyA1), sympy_to_mpmath_matrix(sympyB1), lam, v)
        print(f"  Residual: {res:.2e}")

    print("\n--- TEST 2: Includes infinite eigenvalues ---")
    A_data2 = [
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    B_data2 = [
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

    sympyA2 = sp.Matrix(A_data2)
    sympyB2 = sp.Matrix(B_data2)

    eigvals2, eigvecs2 = generalized_eigen_refine(sympyA2, sympyB2, max_norm=100, prec = 50)
    print("Refined eigenvalues and residuals:")
    for lam, v in zip(eigvals2, eigvecs2):
        print(f"  λ = {complex(float(lam.real), float(lam.imag))}")
        res = compute_residual(sympy_to_mpmath_matrix(sympyA2), sympy_to_mpmath_matrix(sympyB2), lam, v)
        print(f"  Residual: {res:.2e}")

    print("\n--- TEST 3: Near-defective (clustered eigenvalues) ---")
    eps = 1e-12
    A_data3 = [
        [1, 1, 0, 0],
        [0, 1+eps, 0, 0],
        [0, 0, 2, 3],
        [0, 0, -1, 4]
    ]
    B_data3 = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

    sympyA3 = sp.Matrix(A_data3)
    sympyB3 = sp.Matrix(B_data3)

    eigvals3, eigvecs3 = generalized_eigen_refine(sympyA3, sympyB3, max_norm=100, prec = 50)
    print("Refined eigenvalues and residuals:")
    for lam, v in zip(eigvals3, eigvecs3):
        print(f"  λ = {complex(float(lam.real), float(lam.imag))}")
        res = compute_residual(sympy_to_mpmath_matrix(sympyA3), sympy_to_mpmath_matrix(sympyB3), lam, v)
        print(f"  Residual: {res:.2e}")
