import numpy as np
import matplotlib.pyplot as plt
from scipy.special import (
    jv, riccati_jn, legendre_p, legendre_p_all
)
from dataclasses import dataclass, field


@dataclass
class SphericalPlaneWave:
    f_0: complex  # incidient electric field magnitude at the origin point of the plane wave
    k: float  # wave number
    num_samples: int  # number of interpolation points in r
    r_max: float
    N: int  # number of sample points to approximation infinite sum for change of basis
    xz_pairs: np.ndarray = field(init=False)
    yz_pairs: np.ndarray = field(init=False)
    r_theta_pairs: np.ndarray = field(init=False)
    f_inc: np.ndarray = field(init=False, repr=False)
    e_inc: np.ndarray = field(init=False, repr=False)
    h_inc: np.ndarray = field(init=False, repr=False)
    #  f_inc_phi: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        lam = 2 * np.pi / self.k
        self.x = np.linspace(-5 * lam, 5 * lam, num=self.num_samples)
        self.y = np.linspace(-5 * lam, 5 * lam, num=self.num_samples)
        self.z = np.linspace(-5 * lam, 5 * lam, num=self.num_samples)
        self.xz_pairs = np.array([(x_val, z_val) for x_val in self.x for z_val in self.z])
        self.yz_pairs = np.array([(y_val, z_val) for y_val in self.y for z_val in self.z])
        self.r_theta_pairs = np.zeros_like(self.xz_pairs)
        for index, (x, z) in enumerate(self.xz_pairs):
            self.r_theta_pairs[index] = (np.sqrt(x ** 2 + z ** 2), np.atan2(x, z))
        self.e_inc = np.array(
            [
                np.zeros(len(self.r_theta_pairs), dtype=complex),  # e_r
                np.zeros(len(self.r_theta_pairs), dtype=complex),  # e_theta
                np.zeros(len(self.r_theta_pairs), dtype=complex),  # e_phi
            ]
        )
        self.h_inc = np.array(
            [
                np.zeros(len(self.r_theta_pairs), dtype=complex),  # h_r
                np.zeros(len(self.r_theta_pairs), dtype=complex),  # h_theta
                np.zeros(len(self.r_theta_pairs), dtype=complex),  # h_phi
            ]
        )
        self.f_inc = self.construct_excitation()
        self.plot_f_inc()

    def construct_excitation(self):
        f_inc = np.zeros((len(self.r_theta_pairs)), dtype=complex)
        for index, (r, theta) in enumerate(self.r_theta_pairs):
            temp_sum_1 = 0.0
            temp_sum_2 = 0.0
            #temp_sum_f = 0.0
            ricatti_bessels = riccati_jn(self.N, self.k * r)[0]
            legendre_polys, legendre_polys_prime = legendre_p_all(self.N, np.cos(theta), diff_n=1)

            for n in range(self.N):
                temp_sum_1 += (1j ** -n) * (2 * n + 1) * ricatti_bessels[n] * legendre_polys[n]
                temp_sum_2 += (1j ** -n) * (2 * n + 1) * ricatti_bessels[n] * legendre_polys_prime[n]
            # make sure to compute cos(theta) here for theta componenets of e_inc and h_inc
            f_inc[index] = temp_sum_1
            #self.e_inc[0][index] = temp_sum_2
            #self.h_inc[0][index] = temp_sum_2
            #self.e_inc[1][index] = temp_sum_1
            #self.h_inc[1][index] = temp_sum_1
            #self.e_inc[2][index] = temp_sum_1
            #self.h_inc[2][index] = temp_sum_1
        return f_inc

    def plot_f_inc(self):
        title = f"Real Part of Normalized Spherical Plane Wave for N={self.N}"
        X, Y = np.meshgrid(self.x, self.y)
        max_val = max(np.abs(self.f_inc))
        f_inc_cart = np.reshape(self.f_inc, (len(self.x), len(self.y))) / max_val
        plt.contourf(X, Y, f_inc_cart.real)
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
    r_max = 5.0 * np.sqrt(2)
    for n in n_vals:
        SphericalPlaneWave(
            f_0=f_0,
            k=k,
            num_samples=num_samples,
            r_max=r_max,
            N=n
        )
