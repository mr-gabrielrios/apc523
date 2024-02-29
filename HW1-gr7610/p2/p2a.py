# Import numerical analysis packages
import numpy as np, scipy as sp, time
# Import visualization packages
import matplotlib, matplotlib.pyplot as plt
# Change output LaTeX to Computer Modern font
matplotlib.rcParams['mathtext.fontset'] = 'cm'

# Original 2D function
def phi(x, y):
    return np.sin(4*np.pi*x**2)*np.cos(2*np.pi*y**3)

# Exact gradient function
def grad_phi(x, y):
    x_comp = (4*x)*np.cos(4*np.pi*x**2)*np.cos(-2*np.pi*y**3)
    y_comp = (3*y**2)*np.sin(4*np.pi*x**2)*np.sin(-2*np.pi*y**3)
    return 2*np.pi*np.array([x_comp, y_comp])

# 2D discretization
def cds(basis_x, basis_y, in_, h, printout=False):
    lx, ly = len(basis_x), len(basis_y)
    out_x, out_y = np.full(shape=(lx, ly), fill_value=np.nan), np.full(shape=(lx, ly), fill_value=np.nan)
    for j, y in enumerate(basis_y):
        for i, x in enumerate(basis_x):
            if printout:
                print('x-index: {0}, y-index: {1} = ({2:.2f}, {3:.2f})'.format(i, j, x, y))            
            out_x[j, i] = (in_[j, (i-2) % lx] - 8*in_[j, (i-1) % lx] + 8*in_[j, (i+1) % lx] - in_[j, (i+2) % lx])/(12*h)
            out_y[j, i] = (in_[(j-2) % ly, i] - 8*in_[(j-1) % ly, i] + 8*in_[(j+1) % ly, i] - in_[(j+2) % ly, i])/(12*h)
            
    return out_x, out_y

# Error calculation
def err(E, D, h):
    error = 0
    if E[0].shape == D[0].shape and E[1].shape == D[1].shape:
        for k in range(0, len(E)):
            for j, y in enumerate(range(0, E.shape[0])):
                for i, x in enumerate(range(0, E.shape[1])):
                    error += abs(E[k][j, i] - D[k][j, i])**2              
    return np.sqrt(h**2 * error)

# Main program run
def main(h, output_plot=False):
    # Define basis vector ranges
    range_x, range_y = (0, 1), (0, 1)
    # Derive grid step size (assume range_x == range_y)
    N = int((max(range_x) - min(range_x))/h)
    # Define basis vectors
    basis_x, basis_y = [np.linspace(range_x[0], range_x[1], N), 
                        np.linspace(range_y[0], range_y[1], N)]
    # Define grid
    X, Y = np.meshgrid(basis_x, basis_y)
    # Define input data
    in_ = phi(X, Y)
    # Get exact output
    exact = grad_phi(X, Y)
    # Get output approximation
    approx = cds(basis_x, basis_y, in_, h, False)
    # Get error
    error = err(exact, approx, h)
    
    return exact, approx, N, error

if __name__ == "__main__":
    start = time.time()
    errors = {}
    for h in np.logspace(-4, -1, num=25):
        exact, approx, N, error = main(h)
        errors[h] = error
        print('Error at h = {0:.2e} is {1:.2e}'.format(h, error))
        print('Elapsed time: {0:.2f} s'.format(time.time() - start))
        
    fig, ax = plt.subplots(figsize=(4, 2)) 

    ax.plot([k for k in errors.keys()], [v for v in errors.values()], marker='o', lw=2, c='b')
    ax.set_xscale('log')
    ax.set_yscale('log')
     
    ax.set_xlim([min([k for k in errors.keys()]), max([k for k in errors.keys()])])
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim([0, max(errors.values())])
        
    ax.set_xlabel('$h$', fontsize=12);
    ax.set_ylabel('$\| e \|_{L^2}$', fontsize=12);
    
    plt.savefig('p2b.png', dpi=300, bbox_inches='tight')
