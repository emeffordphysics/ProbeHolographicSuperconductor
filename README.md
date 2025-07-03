# Probe Holographic Superconductor Hydrodynamics

This introduces two numerical methods to calculate the hydrodynamics of probe superconductors
as performed in https://arxiv.org/pdf/2212.10410 and https://arxiv.org/pdf/2312.08243.

The first method is a Mathematica notebook. This is fast and adaptable but requires a 
Mathematica license to use.

The second is a Python package. The actual numerics and equations of motion are performed 
in a Jupyter notebook. The equations of motion are derived using sympy. To speed the numerics
up, this notebook first calculates things using double floating point complex
numbers with NumPy and SciPy and then refines the calculations to high precision using
mpmath.

To use the python notebook, make sure that sympy, numpy, scipy, mathmp, and gmpy2 are installed.
