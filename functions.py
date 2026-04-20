import numpy as np
import matplotlib.pyplot as plt

def scale(x):
    a, b = x[0], x[-1]
    return 2 * (x - a) / (b - a) - 1

def unscale(x, a, b):
    return 0.5 * (b - a) * x + 0.5 * (a + b)

def chebyshev_nodes(n):
    k = np.arange(1, n+1)
    x_cheb = np.cos(np.pi * (2*k - 1)/(2*n))
    return x_cheb

def lagrange(x, y, num_points=6, chebyshev=False):
    a, b = x[0], x[-1]

    if chebyshev:
        nodes = chebyshev_nodes(num_points)

    nodes_original = unscale(nodes, a, b) if chebyshev else np.linspace(a, b, num_points)
    idx = np.unique([np.argmin(np.abs(x - xi)) for xi in nodes_original])
    x_uniform = x[idx]
    y_uniform = y[idx]

    x_val = scale(x_uniform)

    x_fine = np.linspace(-1, 1, 1000)
    interp = np.zeros_like(x_fine)
    n = len(x_val)
    for i in range(n):
        terms = np.ones_like(x_fine)
        for j in range(n):
            if i != j:
                terms *= (x_fine - x_val[j]) / (x_val[i] - x_val[j])
        interp += terms * y_uniform[i]

    x_fine = unscale(x_fine, a, b)
    return x_fine, interp, x_uniform, y_uniform

def cubic_spline(x, y, num_points=6, chebyshev=False):
    a, b = x[0], x[-1]

    if chebyshev:
        nodes = chebyshev_nodes(num_points)

    nodes_original = unscale(nodes, a, b) if chebyshev else np.linspace(a, b, num_points)
    idx = np.unique([np.argmin(np.abs(x - xi)) for xi in nodes_original])
    x_uniform = x[idx]
    y_uniform = y[idx]

    x_val = scale(x_uniform)
    n = len(x_val) - 1
    h = np.diff(x_val)

    A = np.zeros((n+1, n+1))
    A[0, 0] = 1
    A[n, n] = 1

    B = np.zeros(n+1)
    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        B[i] = 3 * ((y_uniform[i+1] - y_uniform[i]) / h[i] - (y_uniform[i] - y_uniform[i-1]) / h[i-1])

    a_coef = y_uniform[:-1]
    c_coef = np.linalg.solve(A, B)
    b_coef = (y_uniform[1:] - y_uniform[:-1]) / h - h * (2 * c_coef[:-1] + c_coef[1:]) / 3
    d_coef = (c_coef[1:] - c_coef[:-1]) / (3 * h)

    x_fine = np.linspace(-1, 1, 1000)
    interp = np.zeros_like(x_fine)
    for i in range(n):
        mask = (x_fine >= x_val[i]) & (x_fine < x_val[i+1]) if i < n - 1 else (x_fine >= x_val[i]) & (x_fine <= x_val[i+1])
        dx = x_fine[mask] - x_val[i]
        interp[mask] = a_coef[i] + b_coef[i] * dx + c_coef[i] * dx**2 + d_coef[i] * dx**3

    x_fine = unscale(x_fine, a, b)
    return x_fine, interp, x_uniform, y_uniform
    
def get_interpolation_fig(x, y, title, rows=1, cols=1, figsize=(10, 4)):
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = np.atleast_2d(axs)
    for r in range(rows):
        for c in range(cols):
            ax = axs[r][c]
            ax.plot(x, y, label="Function")
            ax.legend()
            ax.grid(True)

    fig.suptitle(title, fontsize=16)
    fig.supxlabel("Distance")
    fig.supylabel("Elevation")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig, axs

def add_interpolation(ax, x_fine, interp, x_uniform, y_uniform, label, row=1, col=1):
    axis = ax[row-1][col-1]
    line, = axis.plot(x_fine, interp, label=label)
    color = line.get_color()
    axis.scatter(x_uniform, y_uniform, color=color)
    axis.legend()
        
def limit_y(ax, y):
    diff = (max(y) - min(y))/10
    ax.set_ylim(top=max(y)+diff, bottom=min(y)-diff)