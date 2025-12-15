import numpy as np
import matplotlib.pyplot as plt
from scipy.special import (
    jv, jvp, hankel1, hankel2
)
from dataclasses import dataclass, field
from coordinate_transforms import (
    cyl_to_cart
)


@dataclass
class PlaneWave:
    f_0: complex  # incidient electric field magnitude at the origin point of the plane wave
    k: float  # wave number
    num_samples: int  # number of interpolation points in rho
    rho_max: float
    n: int  # number of sample points to approximation infinite sum for change of basis
    xy_pairs: np.ndarray = field(init=False)
    rho_phi_pairs: np.ndarray = field(init=False)
    f_inc_z_cylindrical: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        lam = 2 * np.pi / self.k
        self.x = np.linspace(-5 * lam, 5 * lam, num=self.num_samples)
        self.y = np.linspace(-5 * lam, 5 * lam, num=self.num_samples)
        # self.rho = np.linspace(0, self.rho_max, self.num_rhos)
        # self.phi = np.linspace(0, 2 * np.pi, self.num_phis)
        self.xy_pairs = np.array([(x_val, y_val) for x_val in self.x for y_val in self.y])
        self.rho_phi_pairs = np.zeros_like(self.xy_pairs)
        for index, (x, y) in enumerate(self.xy_pairs):
            self.rho_phi_pairs[index] = (np.sqrt(x ** 2 + y ** 2), np.atan2(y, x))
        self.f_inc_z_cylindrical = self.construct_excitation()

    def construct_excitation(self):
        f_inc_z = np.zeros((len(self.rho_phi_pairs)), dtype=complex)
        for index, (rho, phi) in enumerate(self.rho_phi_pairs):
            inc_sum: complex = 0.0j
            for n in range(-self.n, self.n):
                inc_sum += 1j ** (-n) * jv(n, self.k * rho) * np.exp(1j * n * phi)
            f_inc_z[index] = inc_sum
        f_inc_z = f_inc_z * self.f_0
        return f_inc_z

    def plot_plane_wave(self, title: str = None):
        if title is None:
            title = f"Plane Wave Approximation n={self.n}"
        else:
            title = title
        # x, y, e_inc_z_cart = cyl_to_cart(self.rho, self.phi, self.e_inc_z_cylindrical)
        X, Y = np.meshgrid(self.x, self.y)
        f_inc_z_cart = np.reshape(self.f_inc_z_cylindrical, (len(self.x), len(self.y)))
        plt.contourf(X, Y, f_inc_z_cart.real)
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    n_vals = [1, 5, 10, 20, 40, 80]
    lam = 1
    k = 2 * np.pi / lam
    f_0 = 1.0
    num_samples = 200
    rho_max = 5.0 * np.sqrt(2)
    for n in n_vals:
        plane_wave_approx = PlaneWave(
            f_0=f_0,
            k=k,
            num_samples=num_samples,
            rho_max=rho_max,
            n=n
        )
        plane_wave_approx.plot_plane_wave()

