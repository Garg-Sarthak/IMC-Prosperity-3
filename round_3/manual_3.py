import numpy as np
from scipy.optimize import minimize, differential_evolution
# 1) piecewise CDF F
def F(x):
    """
    Vectorized: x can be scalar or array.
    """
    x = np.asarray(x, dtype=float)
    fx = np.zeros_like(x)

    # piece 1: [160,200)
    m1 = (x >= 160) & (x < 200)
    fx[m1] = (x[m1] - 160) / 110

    # piece 2: [200,250)
    m2 = (x >= 200) & (x < 250)
    fx[m2] = 40 / 110

    # piece 3: [250,320]
    m3 = (x >= 250) & (x <= 320)
    # —— here we assume the intended formula is (x - 210)/110
    fx[m3] = (x[m3] - 210) / 110

    # outside: CDF flat
    fx[x < 160] = 0.0
    fx[x > 320] = 1.0

    return fx

# 2) our objective (we'll minimize its negative)
def negO(vars):
    b1, b2 = vars
    # enforce 160 ≤ b1 ≤ b2 ≤ 320
    if not (160 <= b1 <= b2 <= 320):
        return 1e6
    return - ( F(b1)*(b2 - b1) + F(b2)*(320 - b2) )

# 3) a quick grid search to seed the local solver
grid = np.linspace(160, 320, 161)
B1, B2 = np.meshgrid(grid, grid)
mask = B2 >= B1

Ovals = F(B1)*(B2 - B1) + F(B2)*(320 - B2)
Ovals[~mask] = -np.inf

imax, jmax = np.unravel_index(np.nanargmax(Ovals), Ovals.shape)
x0 = [B1[imax,jmax], B2[imax,jmax]]

# 4) local refine via SLSQP (handles b2>=b1 constraint)
cons = ({'type':'ineq', 'fun': lambda x: x[1]-x[0]})
bnds = [(160,320),(160,320)]
res_loc = minimize(negO, x0, method='SLSQP', bounds=bnds, constraints=cons)
b1_loc, b2_loc = res_loc.x
O_loc = -res_loc.fun

# 5) (optional) global search via Differential Evolution
res_glob = differential_evolution(negO, bounds=bnds, strategy='best1bin', maxiter=1000)
b1_glob, b2_glob = res_glob.x
O_glob = -res_glob.fun

# 6) pick the best
if O_glob > O_loc:
    b1_opt, b2_opt, O_opt = b1_glob, b2_glob, O_glob
else:
    b1_opt, b2_opt, O_opt = b1_loc, b2_loc, O_loc

print(f"→ Optimal b1={b1_opt:.4f}, b2={b2_opt:.4f}, O={O_opt:.4f}")
