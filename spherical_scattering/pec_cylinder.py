import numpy as np
import matplotlib.pyplot as plt
from scipy.special import (
    jv, jvp, h2vp, hankel2, jn
)
from cylindrical_plane_wave import (
    PlaneWave
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

    def s(self, ka: float, phi):
        S_n = np.zeros(self.N, dtype=complex)
        phase_terms = np.zeros(self.N, dtype=complex)
        if self.mode.upper() == "TM":
            for n in range(self.N):
                S_n[n] += jn(n, ka) / hankel2(n, ka)
                phase_terms[n] = np.cos(n * phi)
        elif self.mode.upper() == "TE":
            for n in range(self.N):
                S_n[n] = jvp(n, ka, 1) / h2vp(n, ka, 1)
                phase_terms[n] = np.cos(n * phi)
        else:
            raise ValueError('Mode must be TM or TE')
        # $S(\phi) = S_0 + 2 \sum_{n=1}^{\inf} S_n(ka) * cos(n \phi)
        S = S_n[0] + 2 * np.sum(S_n[1:] * phase_terms[1:], axis=0)
        S_norm = (2.0 / np.pi) * np.abs(S) ** 2
        return S_norm

    def plot_monostatic_scatter_width(self, ax: plt.Axes):
        a_lam = np.linspace(1e-6, 2, 100)
        ka = a_lam * 2 * np.pi
        scatter_width_lambda = np.array([self.s(ka_val, np.pi) for ka_val in ka])
        label = f'{self.mode.upper()} Wave Excitation'
        ax.plot(a_lam, scatter_width_lambda, label=label)
        return ax

    def plot_bistatic_scatter_width(self, ax: plt.Axes):
        phis = np.linspace(0, 2 * np.pi, 100)
        scatter_width_lambda_db = np.array([10 * np.log10(self.s(self.k * self.a, phi)) for phi in phis])
        label = f'{self.mode.upper()} Wave Excitation'
        ax.plot(phis, scatter_width_lambda_db, label=label)
        return ax

if __name__ == '__main__':
    # TM Mode
    pec_cylinder_tm = CylinderPEC("TM")

    # TE Mode
    pec_cylinder_te = CylinderPEC("TE")

    # Plot Monostatic Scatter Width
    fig, ax = plt.subplots()
    pec_cylinder_tm.plot_monostatic_scatter_width(ax)
    pec_cylinder_te.plot_monostatic_scatter_width(ax)
    ax.set_title('2D Monostatic Scatter Width of PEC Cylinder')
    ax.set_xlabel(r'$\frac{a}{\lambda}$')
    ax.set_ylabel(r'$\sigma_{2D}/\lambda$')
    ax.legend()
    plt.show()

    # Plot Bistatic Scatter Width
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    pec_cylinder_tm.plot_bistatic_scatter_width(ax1)
    pec_cylinder_te.plot_bistatic_scatter_width(ax2)
    ax1.set_title(f'2D Bistatic Scatter Width of PEC Cylinder\nTM Wave Excitation')
    ax1.set_xlabel(r'$\phi [radians]$')
    ax1.set_ylabel(r'$\sigma_{2D}/\lambda [dB]$')
    ax2.set_title(f'2D Bistatic Scatter Width of PEC Cylinder\nTE Wave Excitation')
    ax2.set_xlabel(r'$\phi [radians]$')
    ax2.set_ylabel(r'$\sigma_{2D}/\lambda [dB]$')
    plt.show()

