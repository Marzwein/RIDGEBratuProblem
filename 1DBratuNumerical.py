import numpy as np
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
betas_exact = np.arange(0, 8, 0.1)
lambdas_exact = 2*betas_exact*betas_exact/(np.cosh(0.5*betas_exact)*np.cosh(0.5*betas_exact))

#exact solution for a specific x and beta value
def exact(x, beta):
    return 2 * np.log(np.cosh(0.5* beta)/(np.cosh(beta*(x - 0.5))))

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
        
        F_norm = np.linalg.norm(F, ord=np.inf)
        
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

#pseudo-arclength continuation
ds_min = 1e-6
ds_max = 1e-2
ds0 = 1e-4
n_steps = int(2e3)
solv_par = {"tolerance": 1e-10, "param_min" : 0}

contin_result = pycont.arclengthContinuation(CFD, var, lambdas[0], ds_min, ds_max, ds0, n_steps, solver_parameters=solv_par)

#plot bifurcation diagram with (lambda, ||u||_infty)
u_trans = lambda u : np.sign(u[int(N/2)]) * np.max(np.abs(u))
pycont.plotBifurcationDiagram(contin_result, u_transform=u_trans, p_label=r'$\lambda$', u_label=r'$\|u\|_\infty$')

#supremum norm of exact solutions
sups_exact = 2*np.log(np.cosh(0.5* betas_exact)) #x = 0.5 gives the supremum

#plot
plt.plot(lambdas_exact, sups_exact, "--", label=["exact solution"])
plt.plot(succ_lambdas, sup_norms, "-", label=["FD approximation"])
plt.xlabel("λ")
plt.ylabel(r"$\|u\|_\infty$")
plt.xlim(0, 4)
plt.ylim(0,)
plt.title("Bratu bifurcation diagram for a 3-point FD stencil")
plt.grid(True)
plt.show()   