from typing_extensions import deprecated

import numpy as np
from typing import Tuple

@deprecated
def cyl_to_cart(rhos: np.ndarray, phis: np.ndarray, func_rho_phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert rhos.shape == phis.shape
    func_xy = np.zeros((rhos.shape[0], phis.shape[0]), dtype=complex)
    x_array = np.linspace(-5, 5, rhos.shape[0])
    y_array = np.linspace(-5, 5, rhos.shape[0])

    for rho_index, rho in enumerate(rhos):
        for phi_idx, phi in enumerate(phis):
            x_temp = rho * np.cos(phi)
            y_temp = rho * np.sin(phi)
            x_idx = (np.abs(x_array - x_temp)).argmin()
            y_idx = (np.abs(y_array - y_temp)).argmin()
            func_xy[x_idx, y_idx] = func_rho_phi[rho_index, phi_idx]

    return x_array, y_array, func_xy

#def cyl_to_cart_from_pairs()