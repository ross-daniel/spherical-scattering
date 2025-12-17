import numpy as np
import matplotlib.pyplot as plt
from scipy.special import (
    jv, riccati_jn, legendre_p, legendre_p_all
)
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SphericalPlaneWave:
    f_0: Tuple[complex, complex]  # incidient field magnitudes at the origin point of the plane wave [E0, H0]
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
            self.r_theta_pairs[index] = (np.sqrt(x ** 2 + z ** 2), np.atan2(x, z))  # want [0,pi] here --> atan NOT atan2
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

        self.e_inc_phi = self.e_inc[2]
        self.h_inc_phi = self.h_inc[2]

    def construct_excitation(self):
        f_inc = np.zeros((len(self.r_theta_pairs)), dtype=complex)
        for index, (x, z) in enumerate(self.xz_pairs):
            temp_sum_1 = 0.0
            #temp_sum_2 = 0.0
            r = np.sqrt(x ** 2 + z ** 2)
            if r == 0:
                cos_theta = 1.0
            else:
                cos_theta = z / r
            ricatti_bessels = riccati_jn(self.N, self.k * r)[0]
            j_sph = ricatti_bessels / (self.k * r)
            if r == 0:
                j_sph[0] = 1.0
            legendre_polys, legendre_polys_prime = legendre_p_all(self.N, cos_theta, diff_n=1)

            for n in range(self.N):
                temp_sum_1 += (1j ** -n) * (2 * n + 1) * j_sph[n] * legendre_polys[n]
                #temp_sum_2 += (1j ** -n) * (2 * n + 1) * j_sph[n] * legendre_polys_prime[n]
            f_inc[index] = temp_sum_1

            #  Adjust e_inc and h_inc (see Jianming Jin eqs. 7.4.19-7.4.24)
            #  --> e_inc in the H-plane (yz) => phi = 90
            #  --> h_inc in the E-plane (xz) => phi = 0

            #self.e_inc[0][index] = temp_sum_2 --> 0 in H-plane
            #self.h_inc[0][index] = temp_sum_2 --> 0 in E-plane
            #self.e_inc[1][index] = cos_theta * temp_sum_1  --> 0 in H-plane
            #self.h_inc[1][index] = cos_theta * temp_sum_1  --> 0 in E-plane
            self.e_inc[2][index] = temp_sum_1 * -self.f_0[0] # / (self.k * r) --> already done
            self.h_inc[2][index] = temp_sum_1 * self.f_0[1] # / (self.k * r) --> already done
        return f_inc

    def plot_f_inc(self):
        title = f"Real Part of Normalized Spherical Plane Wave for N={self.N}"
        X, Z = np.meshgrid(self.x, self.z)
        #max_val = max(np.abs(self.f_inc))
        f_inc_cart = np.reshape(self.f_inc, (len(self.x), len(self.z))) #/ max_val
        plt.contourf(Z, X, f_inc_cart.real)
        plt.colorbar()
        plt.title(title)
        plt.show()

if __name__ == "__main__":
    n_vals = [1, 5, 10, 20, 40, 80]
    lam = 1
    k = 2 * np.pi / lam
    f_0 = (1.0, 1.0)
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
