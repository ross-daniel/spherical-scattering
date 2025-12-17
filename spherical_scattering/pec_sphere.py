import numpy as np
import matplotlib.pyplot as plt
from fontTools.ufoLib.utils import deprecated
from scipy.special import (
    jv, jn, jvp, hankel1, hankel2, h2vp, legendre_p, legendre_p_all, riccati_jn, spherical_yn, spherical_jn, sph_legendre_p, sph_legendre_p_all
)
from dataclasses import dataclass, field
from typing import Tuple
from coordinate_transforms import (
    cyl_to_cart, plot_cyl_on_cart
)
from spherical_scattering.spherical_plane_wave import SphericalPlaneWave
from constants import *



def riccati_hankel2(n, z, derivative=False):
    jn = spherical_jn(n, z)
    yn = spherical_yn(n, z)
    jn_prime = spherical_jn(n, z, derivative=True)
    yn_prime = spherical_yn(n, z, derivative=True)
    h2 = jn - 1j * yn
    h2_prime = jn_prime - 1j * yn_prime
    if not derivative:
        return z * h2
    return h2 + z * h2_prime

def riccati_hankel2_all(n_max, z, derivative=False):
    n = np.arange(0, n_max + 1)
    jn = spherical_jn(n, z)
    yn = spherical_yn(n, z)
    jn_prime = spherical_jn(n, z, derivative=True)
    yn_prime = spherical_yn(n, z, derivative=True)
    h2 = jn - 1j * yn
    h2_prime = jn_prime - 1j * yn_prime
    if not derivative:
        return z * h2
    else:
        return z * h2, h2 + z * h2_prime

def spherical_legendre_all(n_max, mu):
    p1 = np.zeros(n_max+1, dtype=float)
    d_p1 = np.zeros(n_max+1, dtype=float)

    p1[1] = -1.0
    d_p1[1] = -mu

    for n in range(2, n_max+1):
        p1[n] = ((2*n-1)/(n-1))*mu*p1[n-1] - (n/(n-1))*p1[n-2]
        d_p1[n] = n*mu*p1[n] - (n+1)*p1[n-1]
    return p1, d_p1


class SpherePEC:
    def __init__(self, incident_field: Tuple[complex, complex] = (1.0, 1.0 / eta_0), k: float = 2 * np.pi, a: float = 1.0,
                 N: int = 80, discretization_level: int = 100, max_r: float = 5.0 * np.sqrt(2)):
        self.incident_field = incident_field
        self.k = k
        self.a = a
        self.N = N
        self.num_samples = discretization_level
        self.max_r = max_r
        self.spherical_plane_wave = SphericalPlaneWave(
            f_0=self.incident_field,
            k=self.k,
            num_samples=self.num_samples,
            r_max=self.max_r,
            N=self.N,
        )
        self.x, self.y, self.z = self.spherical_plane_wave.x, self.spherical_plane_wave.y, self.spherical_plane_wave.z
        self.xz_pairs = self.spherical_plane_wave.xz_pairs
        self.yz_pairs = self.spherical_plane_wave.yz_pairs
        self.r_theta_pairs = self.spherical_plane_wave.r_theta_pairs
        self.e_inc_phi = self.spherical_plane_wave.e_inc_phi
        self.h_inc_phi = self.spherical_plane_wave.h_inc_phi

        # find coefficients an and bn
        self.an, self.bn = self.calculate_coefficients()
        # use coefficients an and bn to solve for phi components of electric field in yz plane and magnetic field in xz plane
        #self.e_sc_phi, self.h_sc_phi = self.calculate_scattered_fields()
        # calculate total fields
        #self.e_tot_phi, self.h_tot_phi = self.e_sc_phi + self.e_inc_phi, self.h_sc_phi + self.h_inc_phi
        # plot scattered fields
        #self.plot_scattered_fields()
        # plot total fields
        #self.plot_total_fields_snapshot()
        #self.plot_total_fields_magnitude()

    def calculate_coefficients(self):
        an = np.zeros(self.N, dtype=complex)
        bn = np.zeros(self.N, dtype=complex)
        riccati_bessels, riccati_bessel_primes = riccati_jn(self.N, self.k * self.a)   #spherical_jn(range(self.N), self.k * self.a), spherical_jn(range(self.N), self.k * self.a, derivative=True)
        for n in range(self.N):
            if n == 0:
                continue
            an[n] = -1j ** -n * ((2 * n + 1) * riccati_bessel_primes[n]) / (n * (n+1) * riccati_hankel2(n, self.k * self.a, derivative=True))
            bn[n] = -1j ** -n * ((2 * n + 1) * riccati_bessels[n]) / (n * (n+1) * riccati_hankel2(n, self.k * self.a, derivative=False))
        return an, bn

    @deprecated
    def calc_scattered_fields(self):
        e_sc_phi = np.zeros_like(self.e_inc_phi, dtype=complex)
        h_sc_phi = np.zeros_like(self.h_inc_phi, dtype=complex)
        for index, (x, z) in enumerate(self.xz_pairs):
            e_sum = 0.0
            h_sum = 0.0
            # r = np.sqrt(x ** 2 + z ** 2)
            # cos_theta = np.cos(theta)#z / r
            # if (1e-6 >= theta and -1e-6 <= theta) or (np.pi + 1e-6 >= theta and np.pi-1e-6 <= theta):
            # sin_theta = 1e20
            # else:
            # sin_theta = np.sin(theta)#x / r
            # theta = np.atan(x / z)
            r = np.hypot(x, z)
            mu = z / r
            theta = np.arccos(mu)
            sin_theta = np.sqrt(1.0 - mu * mu)
            cos_theta = mu
            if r > self.a:
                # legendre_polys, legendre_poly_primes, legendre_poly_2primes = legendre_p_all(self.N, cos_theta, diff_n=1) # legendre_poly_2primes
                # print(sph_legendre_p_all(self.N, 1, theta, diff_n=1)[0])
                # all_sph_legendres = sph_legendre_p_all(self.N, 1, theta, diff_n=1)
                # legendre_poly_primes, legendre_poly_2primes = all_sph_legendres[0], all_sph_legendres[1]
                for n in range(1, self.N):
                    h_hat = riccati_hankel2(n, self.k * r, derivative=False)
                    h_hat_prime = riccati_hankel2(n, self.k * r, derivative=True)
                    pn1 = sph_legendre_p(n, 1, theta)[0]
                    if n == 1:
                        d_pn1 = -cos_theta
                    else:
                        d_pn1 = pn1 * cos_theta / sin_theta + sph_legendre_p(n, 2, theta)[0]
                    eterm1 = self.an[n] * 1j * h_hat_prime * pn1 / sin_theta
                    eterm2 = self.bn[n] * h_hat * d_pn1
                    hterm1 = self.an[n] * h_hat * d_pn1
                    hterm2 = self.bn[n] * 1j * h_hat_prime * pn1 / sin_theta
                    e_sum += eterm1 + eterm2
                    h_sum += hterm1 + hterm2

                e_sum = e_sum * self.incident_field[0] / (self.k * r)
                h_sum = -h_sum * self.incident_field[1] / (self.k * r)
            e_sc_phi[index] = e_sum
            h_sc_phi[index] = h_sum
        return e_sc_phi, h_sc_phi

    def calculate_scattered_fields(self):
        e_sc_phi = np.zeros_like(self.e_inc_phi, dtype=complex)
        h_sc_phi = np.zeros_like(self.h_inc_phi, dtype=complex)
        for index, (x, z) in enumerate(self.xz_pairs):
            e_sum = 0.0
            h_sum = 0.0
            r = np.hypot(x, z)
            mu = z / r
            theta = np.arccos(mu)
            sin_theta = np.sqrt(1.0 - mu*mu)
            cos_theta = mu
            if r > self.a:
                # VECTORIZED OPERATIONS
                n = np.arange(1, self.N)
                z = self.k * r
                h_hat, h_hat_prime = riccati_hankel2_all(self.N-1, z, derivative=True)
                p1, d_p1 = spherical_legendre_all(self.N-1, mu)

                e_sum = np.sum(self.an[n] * 1j * h_hat_prime[n] * p1[n] + self.bn[n] * h_hat[n] * d_p1[n])
                h_sum = np.sum(self.an[n] * h_hat[n] * d_p1[n] + self.bn[n] * 1j * h_hat_prime[n] * p1[n])

                e_sum = e_sum * self.incident_field[0] / (self.k * r)
                h_sum = -h_sum * self.incident_field[1] / (self.k * r)
            e_sc_phi[index] = e_sum
            h_sc_phi[index] = h_sum
        return e_sc_phi, h_sc_phi

    def plot_fields(self, func_zy, func_zx, title1 = None, title2 = None):
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        if title1 is None:
            title1 = r'Real Part of Scattered Electric Field ($\phi$-component)'
        else:
            title1 = title1
        if title2 is None:
            title2 = r'Real Part of Scattered Magnetic Field ($\phi$-component)'
        plot_cyl_on_cart(
            fig=fig1,
            ax=ax1,
            x_array=self.z,
            y_array=self.y,
            func_xy=func_zy,
            title=title1,
            xlabel='z',
            ylabel='y'
        )
        plot_cyl_on_cart(
            fig=fig2,
            ax=ax2,
            x_array=self.z,
            y_array=self.x,
            func_xy=func_zx,
            title=title2,
            xlabel='z',
            ylabel='x'
        )
        ax1.add_patch(plt.Circle((0, 0), self.a, fill=True, color='white'))
        ax2.add_patch(plt.Circle((0, 0), self.a, fill=True, color='white'))
        plt.show()

    def plot_scattered_fields(self):
        title1 = r'Real Part of Scattered Electric Field ($\phi$-component)'
        title2 = r'Real Part of Scattered Magnetic Field ($\phi$-component)'
        func_zy = self.e_sc_phi.real
        func_zx = self.h_sc_phi.real * eta_0
        self.plot_fields(func_zy, func_zx, title1, title2)

    def plot_total_fields_snapshot(self, title1 = None, title2 = None):
        title1 = r'Real Part of Total Electric Field ($\phi$-component)'
        title2 = r'Real Part of Total Magnetic Field ($\phi$-component)'
        func_zy = self.e_tot_phi.real
        func_zx = self.h_tot_phi.real * eta_0
        self.plot_fields(func_zy, func_zx, title1, title2)

    def plot_total_fields_magnitude(self, title1 = None, title2 = None):
        title1 = r'Magnitude of Total Electric Field ($\phi$-component)'
        title2 = r'Magnitude of Total Magnetic Field ($\phi$-component)'
        func_zy = np.abs(self.e_tot_phi)
        func_zx = np.abs(self.h_tot_phi * eta_0)
        self.plot_fields(func_zy, func_zx, title1, title2)

    def monostatic_rcs(self, ka):
        nmax = int(np.ceil(ka + 4.0 * ka ** (1.0 / 3.0) + 2.0))
        nmax = max(1, min(self.N, nmax))
        n = np.arange(1, nmax+1)
        h_hat, h_hat_prime = riccati_hankel2_all(self.N, ka, derivative=True)
        S = np.sum(((-1) ** n) * (2*n + 1) / h_hat_prime[n] / h_hat[n])
        S_norm = np.abs(S) ** 2
        return S_norm / (ka ** 2)

    def monostatic_rcs_chat(self, ka):
        ka = float(ka)

        # truncate the series based on ka (prevents huge high-order terms for small ka)
        n_max = min(self.N, int(np.ceil(ka + 4.0 * ka ** (1.0 / 3.0) + 2.0)))
        if n_max < 1:
            return 0.0

        h_hat, h_hat_prime = riccati_hankel2_all(n_max, ka, derivative=True)

        n = np.arange(1, n_max + 1)
        signs = np.where(n % 2 == 0, 1.0, -1.0)  # (-1)^n without precedence bugs

        # Eq. 7.4.48 uses 1/(Hhat' * Hhat). Do it as two divides to avoid overflow in the product.
        term = signs * (2.0 * n + 1.0) / h_hat_prime[n] / h_hat[n]
        S = np.sum(term)

        return (np.abs(S) ** 2) / (ka ** 2)

    def plot_monostatic_rcs_db(self, ax: plt.Axes):
        a = np.linspace(1e-2, 3.0, 300)
        rcs_vals = np.array([self.monostatic_rcs(2*np.pi*a_val) for a_val in a])
        ax.plot(a, 10 * np.log10(rcs_vals))
        return ax

def main():
    pec_sphere = SpherePEC()
    fig, ax = plt.subplots()
    pec_sphere.plot_monostatic_rcs_db(ax)
    ax.set_title('Monostatic RCS of PEC Sphere Excited by TM Wave')
    ax.set_xlabel(r"$\frac{a}{\lambda}$")
    ax.set_ylabel(r"$\sigma_{3D}/\pi a^2$")
    plt.show()

if __name__ == '__main__':
    main()