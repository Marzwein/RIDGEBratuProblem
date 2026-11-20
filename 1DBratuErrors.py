import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

#N = int(1e3) #number of space discretization points

beta = 1
lam = 2*beta**2/(np.cosh(0.5*beta)**2)

#Multiple N's for error visualization
N_values = [20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]
h_values = np.array([])
errors = np.array([])

#exact solution for a specific x and beta value
def exact(x, beta):
    return 2 * np.log(np.cosh(0.5* beta)/(np.cosh(beta*(x - 0.5))))

#Central finite differences
def CFD(previous, lam, N_int):
    #Take previous values and u(0) = u(1) = 0
    u_values = np.zeros(N_int + 1) 
    u_values[1:-1] = previous 
    
    #compute the central finite difference
    Delta_u = (u_values[:-2] - 2*u_values[1:-1] + u_values[2:])/(h**2)
    
    #return Delta u - lambda e^u = 0
    return Delta_u + lam*np.exp(u_values[1:-1])

#Newton solver
def Newton(var, lam, N_int, tol = 1e-10, max_itercount = 100):

    for i in range(max_itercount):
        F = CFD(var, lam, N_int)
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
        off_diag = np.full(N_int - 2, 1/h**2) #off_diagonals of 1/h^2
        J = diags([off_diag, diag, off_diag], offsets=[-1, 0, 1], format="csc")
        
        #solve the equations in Newton's method
        d = spsolve(J, -F)
        #var += d
        
        #u = np.zeros(N + 1)
        #u[1:-1] = var   
        
        #damping
        alpha = 1.0
        
        #try to solve for var += d, if it doesnt work, damp d with factor alpha
        while alpha > 1e-6: 
            try_var = var + alpha*d     
            
            F_try = CFD(try_var, lam, N_int)
            
            #if the norm is lower than the F_norm, we are done
            if np.linalg.norm(F_try, ord=np.inf) < F_norm:
                var = try_var
                break
            
            alpha *= 0.5
        else:
            print("Newton step failed; reduce the lambda step or change the initial guess.")
            return var, False
    
    return var, False

###################################################################

###################################################################

for N_int in N_values:
    h = 1/N_int

    x_grid = np.linspace(0, 1, N_int+1)
    
    #exact solution stuff
    betas_exact = np.arange(0, 8, 0.1)
    lambdas_exact = 2*betas_exact**2/(np.cosh(0.5*betas_exact)**2)

    #inital guess based on e^u ~ 1 for small lambda
    #use another initial guess for large(r) starting lambda
    var = 0.5*lam*x_grid[1:-1]*(1 - x_grid[1:-1]) 
    sup_norms = []
    
    #initial guess for upper solution
    var_2 = 5*np.sin(np.pi*x_grid[1:-1])
    
    sol = var #change initial guess between var & var_2
    
    #sup_norm
    u, conv = Newton(sol, lam, N_int, tol=1e-8)
        
    if not conv:
        print(f"Newton solver failed for N = {N_int}")
    
    u_num = np.zeros(N_int + 1)
    u_num[1:-1] = u
    
    #supremum norm of exact solutions
    u_exact = exact(x_grid, beta)
    
    #maximum of all the errors in the norm
    err = np.max(np.abs(u_num - u_exact))
    errors = np.append(errors, err)
    h_values = np.append(h_values, h)
    
    print(f"N = {N_int}, h = {h:.6f}, error = {err:.6e}")

#convergence
conv_orders = np.log(errors[:-1]/errors[1:]) / np.log(h_values[:-1]/h_values[1:])

#plot for the errors
plt.figure(figsize=(8,6))
plt.loglog(h_values, errors, 'o-', label="CFD error")
plt.loglog(h_values, h_values, '--', label="error of order 1")
plt.loglog(h_values, np.array(h_values)**2, '--', label="error of order 2")

plt.xlabel("h")
plt.ylabel(r"$||u_{CFD}-u_{exact}||_\infty$")
plt.title("Convergence of the finite difference method")
plt.legend()
plt.show()
