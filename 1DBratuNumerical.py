import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import pycont

N = int(1e2) #number of space discretization points
h = 1/N #space stepsize
x_grid = np.linspace(0, 1, N+1)

#dlambda = 1/K
K = 500
lambdas = np.linspace(0, 3.5, K)

#exact solution stuff
betas_exact = np.arange(0, 10, 0.1)
lambdas_exact = 2*betas_exact*betas_exact/(np.cosh(0.5*betas_exact)*np.cosh(0.5*betas_exact))

#exact solution for a specific x and beta value
def exact(x_grid, betas):
    return 2 * np.log(np.cosh(0.5* betas)/(np.cosh(betas*(x_grid - 0.5))))

#Central finite differences
def CFD(previous, lam):
    #Take previous values and u(0) = u(1) = 0
    u_values = np.zeros(N + 1, dtype=float) 
    u_values[1:-1] = previous 
    
    #compute the central finite difference
    Delta_u = (u_values[:-2] - 2*u_values[1:-1] + u_values[2:])/(h*h)
    
    #return Delta u - lambda e^u = 0
    return Delta_u + lam*np.exp(u_values[1:-1])

#Newton solver
def Newton(var, lam, tol = 1e-6, max_itercount = 100):
    F = CFD(var, lam)
    for i in range(max_itercount):
        
        F_norm = lg.norm(F, ord=np.inf)
        
        #check whether ||u^{i+1} - u^i||_\infty < tol
        if F_norm < tol:
            print(f"It has converged in {i} iterations.")
            return var, True
        # F_norm > 10: #if norm diverges, break the loop
        #    print("Newton step failed; norm diverging to infinity")
        #    return var, False
        
        #Jacobian stuff
        diag = -2/h**2 + lam*np.exp(var) #diagonal of -2/h^2 +lambda e^u_i
        off_diag = np.full(N - 2, 1/h**2) #off_diagonals of 1/h^2
        J = diags([off_diag, diag, off_diag], offsets=[-1, 0, 1], format="csc")
        
        #solve the equations in Newton's method
        d = spsolve(J, -F)

        #damping
        alpha = 1.0
        
        #try to solve for var += d, if it doesnt work, damp d with factor alpha
        while alpha > 1e-6: 
            try_var = var + alpha*d     
            
            F_try = CFD(try_var, lam)
            
            #if the norm is lower than the F_norm, we are done
            if np.linalg.norm(F_try, ord=np.inf) < F_norm:
                var = try_var
                F = F_try
                break
            
            alpha *= 0.5
    else:
        print("Newton step failed; reduce the lambda step or change the initial guess.")
        return var, False
    
    return var, False

#inital guess based on e^u ~ 1 for small lambda
#use another initial guess for large(r) starting lambda
#var = 0.5*lambdas[0]*x_grid[1:-1]*(1 - x_grid[1:-1]) 
var = np.zeros(N-1, dtype=float)
sup_norms = []

var_2 = 5*np.sin(np.pi*x_grid[1:-1])

succ_lambdas = []
sol = var #change initial guess between var & var_2

#contain sup_norms for the lambdas for which there is a solution.
for lam in lambdas:
    u, conv = Newton(sol, lam)
    
    if not conv:
        break
    
    #parameter continuation, use previous solution as guess for next lambda
    sol = u 
    sup_norms.append(np.max(u))
    succ_lambdas.append(lam)

#supremum norm of exact solutions
sups_exact = 2*np.log(np.cosh(0.5* betas_exact)) #x = 0.5 gives the supremum

#pseudo-arclength continuation
ds_min = 1e-6
ds_max = 1e-2
ds0 = 1e-4
n_steps = int(3200)
solv_par = {"tolerance": 1e-10, "param_min" : 0}

#get initial 'upper' solution on the sol curve for specific lambda
var_3, conv = Newton(var_2, 0.5)

#needs a precise solution instead of a mediocre initial guess as input
contin_result = pycont.arclengthContinuation(CFD, var_3, 0.5, ds_min, ds_max, ds0, n_steps, solver_parameters=solv_par)
# brs_num = len(contin_result.branches)

#plot bifurcation diagram with (lambda, ||u||_infty)
u_trans = lambda u : np.sign(u[int(N/2)]) * np.max(np.abs(u))
# plt.plot(lambdas_exact, sups_exact, "--", label=["exact solution"])
# pycont.plotBifurcationDiagram(contin_result, u_transform=u_trans, p_label=r'$\lambda$', u_label=r'$\|u\|_\infty$')

#plot bifurcation diagram with (lambda, ||u||_infty)
fig, ax = plt.subplots()

ax.plot(lambdas_exact, sups_exact, "b--", label="exact solution")

for i, branch in enumerate(contin_result.branches):
    lam_num = branch.p_path
    sup_u_num = np.max(branch.u_path, axis=1)
    ax.plot(lam_num, sup_u_num, "r-",
            label="computed solution" if i == 0 else None)
    
#colors for the different kinds of points 
style = {"SP" : 'go', "LP" : 'ko', "BP" : 'ro', "HB": 'mo', "DSFLOOR": 'co'}
for event in contin_result.events:
    if event.kind == "LP":
        ax.plot(event.p, u_trans(event.u), style[event.kind], label="Fold")

ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\|u\|_{\infty}$")
plt.xlim(0, 4)
plt.ylim(0,)
plt.title("Bratu bifurcation diagram for a 3-point FD stencil")
ax.legend()





#plot
# plt.plot(lambdas_exact, sups_exact, "--", label=["exact solution"])
# plt.plot(succ_lambdas, sup_norms, "-", label=["FD approximation"])
# plt.xlabel("λ")
# plt.ylabel(r"$\|u\|_\infty$")
# plt.xlim(0, 4)
# plt.ylim(0,)
# plt.title("Bratu bifurcation diagram for a 3-point FD stencil")
# plt.grid(True)
# plt.show()   
