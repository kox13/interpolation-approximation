from functions import *
from pathlib import Path
import pandas as pd
import sys

files = [Path("paths/genoa_rapallo.txt"), Path("paths/Obiadek.csv"), 
         Path("paths/chelm.txt")]

paths = []
for file in files:
    data = pd.read_csv(file)
    data.columns = ['distance', 'elevation']
    paths.append((file.stem, data))

# lagrange
for path in paths:
    distance = np.array(path[1]['distance'].values)
    elevation = np.array(path[1]['elevation'].values)

    fig, axs = get_interpolation_fig(distance, elevation, f"Lagrange Interpolation for {path[0]}", rows=3, cols=2, figsize=(10, 12))
    x_fine, interp, x_uniform, y_uniform = lagrange(distance, elevation, 6)
    add_interpolation(axs, x_fine, interp, x_uniform, y_uniform, label=f"n = 6", row=1, col=1)
    
    x_fine, interp, x_uniform, y_uniform = lagrange(distance, elevation, 9)
    add_interpolation(axs, x_fine, interp, x_uniform, y_uniform, label=f"n = 9", row=1, col=2)
    
    x_fine, interp, x_uniform, y_uniform = lagrange(distance, elevation, 12)
    add_interpolation(axs, x_fine, interp, x_uniform, y_uniform, label=f"n = 12", row=2, col=1)
    
    x_fine, interp, x_uniform, y_uniform = lagrange(distance, elevation, 15)
    add_interpolation(axs, x_fine, interp, x_uniform, y_uniform, label=f"n = 15", row=2, col=2)
    
    x_fine, interp, x_uniform, y_uniform = lagrange(distance, elevation, 18)
    add_interpolation(axs, x_fine, interp, x_uniform, y_uniform, label=f"n = 18", row=3, col=1)
    limit_y(axs[2][0], elevation)
    
    x_fine, interp, x_uniform, y_uniform = lagrange(distance, elevation, 21)
    add_interpolation(axs, x_fine, interp, x_uniform, y_uniform, label=f"n = 21", row=3, col=2)
    limit_y(axs[2][1], elevation)

    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        plt.savefig(fname=f"charts/lagrange_{path[0]}.png")
    else:
        plt.show()
    
    fig, axs = get_interpolation_fig(distance, elevation, f"Lagrange Interpolation for {path[0]} with Chebyshev nodes", rows=1, cols=2, figsize=(12,4))
    x_fine, interp, x_uniform, y_uniform = lagrange(distance, elevation, 60, chebyshev=True)
    add_interpolation(axs, x_fine, interp, x_uniform, y_uniform, label=f"n = 60", row=1, col=1)
    
    x_fine, interp, x_uniform, y_uniform = lagrange(distance, elevation, 100, chebyshev=True)
    add_interpolation(axs, x_fine, interp, x_uniform, y_uniform, label=f"n = 100", row=1, col=2)
    limit_y(axs[0][1], elevation)

    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        plt.savefig(fname=f"charts/lagrange_{path[0]}_chebyshev.png")
    else:
        plt.show()

# cubic spline
for path in paths:
    distance = np.array(path[1]['distance'].values)
    elevation = np.array(path[1]['elevation'].values)

    fig, axs = get_interpolation_fig(distance, elevation, f"Cubic Spline Interpolation for {path[0]}", rows=3, cols=2, figsize=(10, 12))
    x_fine, interp, x_uniform, y_uniform = cubic_spline(distance, elevation, 6)
    add_interpolation(axs, x_fine, interp, x_uniform, y_uniform, label=f"n = 6", row=1, col=1)
    
    x_fine, interp, x_uniform, y_uniform = cubic_spline(distance, elevation, 10)
    add_interpolation(axs, x_fine, interp, x_uniform, y_uniform, label=f"n = 10", row=1, col=2)
    
    x_fine, interp, x_uniform, y_uniform = cubic_spline(distance, elevation, 20)
    add_interpolation(axs, x_fine, interp, x_uniform, y_uniform, label=f"n = 20", row=2, col=1)
    
    x_fine, interp, x_uniform, y_uniform = cubic_spline(distance, elevation, 30)
    add_interpolation(axs, x_fine, interp, x_uniform, y_uniform, label=f"n = 30", row=2, col=2)
    
    x_fine, interp, x_uniform, y_uniform = cubic_spline(distance, elevation, 50)
    add_interpolation(axs, x_fine, interp, x_uniform, y_uniform, label=f"n = 50", row=3, col=1)
    limit_y(axs[2][0], elevation)
    
    x_fine, interp, x_uniform, y_uniform = cubic_spline(distance, elevation, 1000)
    add_interpolation(axs, x_fine, interp, x_uniform, y_uniform, label=f"n = 100", row=3, col=2)
    limit_y(axs[2][1], elevation)

    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        plt.savefig(fname=f"charts/spline_{path[0]}.png")
    else:
        plt.show()