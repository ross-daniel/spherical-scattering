import numpy as np
import matplotlib.pyplot as plt
from scipy.special import (
    jv, hankel1, hankel2
)
from dataclasses import dataclass, field


@dataclass
class PLANE_WAVE_EXCITATION:
    e_0: complex  # incidient electric field magnitude at the origin point of the plane wave
    k: float  # wave number
    num_rhos: int  # number of interpolation points in rho
    rho_max: float
    num_phi: int  # number of interpolation points in phi
    n: int  # number of sample points to approximation infinite sum for change of basis
    rho: np.ndarray = field(init=False)
    phi: np.ndarray = field(init=False)
    e_inc_z_cylindrical: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.rho = np.linspace(1e-9, self.rho_max, self.num_rhos)
        self.phi = np.linspace(0, 2 * np.pi, self.num_phi)

    def construct_cartesian_excitation(self):
        e_inc_z = np.zeros((self.num_rhos, self.num_phi), dtype=complex)
        for rho_index, rho in enumerate(self.rho):
            for phi_index, phi in enumerate(self.phi):
                inc_sum: complex = 0.0j
                for n in range(-int(self.n/2), int(self.n/2)):
                    inc_sum += 1j ** (-n) * jv(n, self.k * rho) * np.exp(1j * n * phi)
                e_inc_z[rho_index, phi_index] = inc_sum
        return e_inc_z