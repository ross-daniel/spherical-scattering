import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

def cyl_to_cart(rho_phi: np.ndarray) -> np.ndarray:
    if type(rho_phi[0]) is not Tuple[float, float]:
        raise("Expected rho_phi to be a list a tuples containg 2 float values each")
    xy_list = [(np.sqrt(x ** 2 + y ** 2), np.atan2(y, x)) for x, y in rho_phi]
    return np.asarray(xy_list)

def plot_cyl_on_cart(fig: plt.Figure, ax: plt.Axes, x_array: np.ndarray, y_array: np.ndarray, func_xy: np.ndarray, title = None) -> None:
    f_xy = np.reshape(func_xy, (len(x_array), len(y_array)))
    X, Y = np.meshgrid(x_array, y_array)
    cs = ax.contourf(X, Y, f_xy)
    if title is None:
        title = "Field"
    else:
        title = title
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(cs, ax=ax)