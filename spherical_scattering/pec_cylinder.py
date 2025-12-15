import numpy as np
import matplotlib.pyplot as plt
from scipy.special import (
    jv, jvp, h2vp, hankel2
)
import pandas as pd
from cylindrical_plane_wave import (
    PLANE_WAVE_EXCITATION_TM, PLANE_WAVE_EXCITATION_TE, PlaneWave
)
from coordinate_transforms import (
    cyl_to_cart, plot_cyl_on_cart
)


class CylinderPEC:
    def __init__(self, mode: str = "TM", incident_field: complex = 1.0, k: float = 2 * np.pi, a: float = 1.0,
                 N: int = 40, discretization_level: int = 100, max_rho: float = 5.0 * np.sqrt(2)):
        if mode.upper() != "TM" and mode.upper() != "TE":
            raise ValueError("Invalid mode, must be 'TM' or 'TE'")
        # initialize variables
        self.mode = mode
        self.f_inc0 = incident_field
        self.k = k
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
        self.an = self.calculate_coefficients()  # get coefficients of the scattered field
        self.f_scz = self.compute_scattered_field()  # get the scattered field itself
        self.f_tot = self.f_incz.f_inc_z_cylindrical + self.f_scz  # total field is the sum of scattered and incident
        # plot
        self.plot_scattered_field()
        self.plot_total_field()

    def calculate_coefficients(self):
        an = np.zeros(2 * self.N, dtype=complex)
        if self.mode == "TM":
            for n in range(-self.N, self.N):
                index = n + self.N
                an[index] = - (self.f_inc0 * 1j ** (-n) * jv(n, self.k * self.a)) / (hankel2(n, self.k * self.a))
        elif self.mode == "TE":
            for n in range(-self.N, self.N):
                index = n + self.N
                an[index] = - (self.f_inc0 * 1j ** (-n) * jvp(n, self.k * self.a, 1)) / h2vp(n, self.k * self.a, 1)
        return an

    def compute_scattered_field(self):
        f_scz = np.zeros(len(self.rho_phi_pairs), dtype=complex)
        for index, (rho, phi) in enumerate(self.rho_phi_pairs):
            sc_sum: complex = 0.0
            if rho <= self.a:
                f_scz[index] = 0.0
            else:
                for n in range(-self.N, self.N):
                    n_index = n + self.N
                    sc_sum += self.an[n_index] * hankel2(n, self.k * rho) * np.exp(1j * n * phi)
                f_scz[index] = sc_sum
        return f_scz

    def plot_scattered_field(self):
        fig, ax = plt.subplots()
        if self.mode.upper() == "TM":
            title = "Scattered Electric Field from\nTM Plane Wave Incident on PEC Cylinder"
        elif self.mode.upper() == "TE":
            title = "Scattered Magnetic Field from\nTE Plane Wave Incident on PEC Cylinder"
        else:
            title = None
        plot_cyl_on_cart(
            fig=fig,
            ax=ax,
            x_array=self.x,
            y_array=self.y,
            func_xy=self.f_scz.real,
            title=title
        )
        circ = plt.Circle((0, 0), self.a, fill=True, color='gray')
        ax.add_patch(circ)
        plt.show()

    def plot_total_field(self):
        fig, ax = plt.subplots()
        if self.mode.upper() == "TM":
            title = "Total Electric Field from\nTM Plane Wave Incident on PEC Cylinder"
        elif self.mode.upper() == "TE":
            title = "Total Magnetic Field from\nTE Plane Wave Incident on PEC Cylinder"
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
        circ = plt.Circle((0, 0), self.a, fill=True, color='gray')
        ax.add_patch(circ)
        plt.show()

if __name__ == '__main__':
    # TM Mode
    pec_cylinder_tm = CylinderPEC("TM")

    # TE Mode
    pec_cylinder_te = CylinderPEC("TE")

