import numpy as np
import matplotlib.pyplot as plt
from scipy.special import (
    jv, jvp, h2vp, hankel2, hankel1
)
from cylindrical_plane_wave import (
    PlaneWave
)
from coordinate_transforms import (
    cyl_to_cart, plot_cyl_on_cart
)
import spherical_scattering.constants as const


class DielectricCylinder:
    def __init__(self, material: tuple[float, float], mode: str = "TM", incident_field: complex = 1.0, k: float = 2 * np.pi, a: float = 1.0,
                 N: int = 40, discretization_level: int = 100, max_rho: float = 5.0 * np.sqrt(2)):
        if mode.upper() != "TM": #and mode.upper() != "TE":
            raise ValueError("Invalid mode, must be 'TM'")
        # initialize variables
        self.mode = mode
        self.eps_r, self.mu_r = material[0], material[1]
        self.f_inc0 = incident_field
        self.k = k
        self.k_d = np.sqrt(self.eps_r) * self.k
        self.omega = self.k * const.c_0
        self.a = a
        self.N = N
        self.num_samples = discretization_level
        self.max_rho = max_rho
        # calculate other parameters
        self.f_incz = PlaneWave(
            f_0=self.f_inc0,
            k=self.k,
            num_samples=self.N,
            rho_max=self.max_rho,
            n=self.num_samples,
        )
        self.x = self.f_incz.x
        self.y = self.f_incz.y
        self.xy_pairs = self.f_incz.xy_pairs
        self.rho_phi_pairs = self.f_incz.rho_phi_pairs
        # solve system
        self.an, self.cn = self.calculate_coefficients()  # get coefficients of the scattered field
        self.f_intz = self.compute_internal_field()  # compute the field inside the cylinder
        self.f_scz = self.compute_scattered_field()  # get the scattered field itself
        self.f_tot = self.f_incz.f_inc_z_cylindrical + self.f_scz + self.f_intz  # total field is the sum of scattered and incident
        # plot
        self.plot_scattered_field()
        self.plot_total_field()

    def calculate_coefficients(self):
        an = np.zeros(2 * self.N, dtype=complex)
        cn = np.zeros(2 * self.N, dtype=complex)
        r = self.a * self.k
        r_d = self.a * self.k_d
        for n in range(-self.N, self.N):
            index = n + self.N
            denom = np.sqrt(self.mu_r) * h2vp(n, r, 1) * jv(n, r_d) - np.sqrt(self.mu_r) * hankel2(n, r) * jvp(n, r_d, 1)
            num = np.sqrt(self.mu_r) * jvp(n, r) * jv(n, r_d) - np.sqrt(self.eps_r) * jv(n, r) * jvp(n, r_d, 1)
            an[index] = -(1j ** (-n)) * num / denom
            cn[index] = (1j ** (-(n+1)) / (np.pi * r)) * 2 * np.sqrt(self.mu_r) / denom
        return an, cn

    def compute_internal_field(self):
        f_int = np.zeros(len(self.rho_phi_pairs), dtype=complex)
        for index, (rho, phi) in enumerate(self.rho_phi_pairs):
            if rho > self.a:
                f_int[index] = 0.0
            else:
                temp_sum = 0.0
                for n in range(-self.N, self.N):
                    n_index = n + self.N
                    temp_sum += self.cn[n_index] * jv(n, self.k_d * rho) * np.exp(1j * n * phi)
                f_int[index] = self.f_inc0 * temp_sum  #self.f_inc0 * temp_sum / (1j * np.sqrt(self.mu_r * const.mu_0 / (self.eps_r * const.eps_0)))
        return f_int

    def compute_scattered_field(self):
        f_scz = np.zeros(len(self.rho_phi_pairs), dtype=complex)
        for index, (rho, phi) in enumerate(self.rho_phi_pairs):
            if rho <= self.a:
                f_scz[index] = 0.0
            else:
                temp_sum = 0.0
                for n in range(-self.N, self.N):
                    n_index = n + self.N
                    temp_sum += self.an[n_index] * hankel2(n, self.k * rho) * np.exp(1j * n * phi)
                f_scz[index] = self.f_inc0 * temp_sum
        return f_scz

    def plot_scattered_field(self):
        fig, ax = plt.subplots()
        if self.mode.upper() == "TM":
            title = "Scattered Electric Field from\nTM Plane Wave Incident on Dielectric Cylinder"
        elif self.mode.upper() == "TE":
            title = "Scattered Magnetic Field from\nTE Plane Wave Incident on Dielectric Cylinder"
        else:
            title = None
        plot_cyl_on_cart(
            fig=fig,
            ax=ax,
            x_array=self.x,
            y_array=self.y,
            func_xy=self.f_scz.real + self.f_intz.real,
            title=title
        )
        circ = plt.Circle((0, 0), self.a, fill=False, color='gray')
        ax.add_patch(circ)
        plt.show()

    def plot_total_field(self):
        fig, ax = plt.subplots()
        if self.mode.upper() == "TM":
            title = "Total Electric Field from\nTM Plane Wave Incident on Dielectric Cylinder"
        elif self.mode.upper() == "TE":
            title = "Total Magnetic Field from\nTE Plane Wave Incident on Dielectric Cylinder"
        else:
            title = None
        plot_cyl_on_cart(
            fig=fig,
            ax=ax,
            x_array=self.x,
            y_array=self.y,
            func_xy=self.f_tot.real,
            title=title
        )
        circ = plt.Circle((0, 0), self.a, fill=False, color='gray')
        ax.add_patch(circ)
        plt.show()

    def s(self, ka, phi):
        S_n = np.zeros(self.N, dtype=complex)
        phase_terms = np.zeros_like(S_n)
        kda = ka * np.sqrt(self.eps_r * self.mu_r)
        for n in range(self.N):
            # compute S_n[n] and phase_term[n]
            num = np.sqrt(self.mu_r) * jvp(n, ka, 1) * jv(n, kda) - np.sqrt(self.eps_r) * jv(n, ka) * jvp(n, kda, 1)
            denom = np.sqrt(self.mu_r) * h2vp(n, ka, 1) * jv(n, kda) - np.sqrt(self.eps_r) * hankel2(n, ka) * jvp(n, kda, 1)
            S_n[n] = num / denom
            phase_terms[n] = np.cos(n * phi)
        # $S(\phi) = S_0 + 2 \sum_{n=1}^{\inf} S_n(ka) * cos(n \phi)
        S = S_n[0] + 2 * np.sum(S_n[1:] * phase_terms[1:], axis=0)
        S_norm = (2.0 / np.pi) * np.abs(S) ** 2
        return S_norm

    def plot_bistatic_scatter_width(self, ax: plt.Axes):
        phis = np.linspace(0, 2 * np.pi, 100)
        scatter_width_lambda_db = np.array([10 * np.log10(self.s(self.k * self.a, phi)) for phi in phis])
        label = f'{self.mode.upper()} Wave Excitation'
        ax.plot(phis, scatter_width_lambda_db, label=label)
        return ax

if __name__ == "__main__":
    dielectric_cylinder = DielectricCylinder((4.0, 1.0))

    # Plot Bistatic Scatter Width
    fig1, ax1 = plt.subplots()
    dielectric_cylinder.plot_bistatic_scatter_width(ax1)
    ax1.set_title(f'2D Bistatic Scatter Width of Dielectric Cylinder\nTM Wave Excitation')
    ax1.set_xlabel(r'$\phi [radians]$')
    ax1.set_ylabel(r'$\sigma_{2D}/\lambda [dB]$')
    plt.show()