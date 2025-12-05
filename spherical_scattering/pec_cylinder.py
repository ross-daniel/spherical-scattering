import numpy as np
import matplotlib.pyplot as plt
from scipy.special import (
    jv, hankel1, hankel2
)
from scipy.interpolate import RegularGridInterpolator
import plotly.express as px
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import pandas as pd


class PEC_CYLINDER_TM:
    def __init__(self, e_incident0: complex, k: float, a: float = 1, num_samples_n: int = 40,
                 num_samples_phi: int = 100, num_samples_rho: int = 100, max_rho: float = 5.0 * np.sqrt(2)):
        self.e_inc0 = e_incident0
        self.num_samples_n = num_samples_n
        self.k = k
        self.a = a
        self.num_samples_phi = num_samples_phi
        self.phi = np.linspace(0, 2*np.pi, self.num_samples_phi)
        self.num_samples_rho = num_samples_rho
        self.max_rho = max_rho
        self.rho = np.linspace(0, self.max_rho, self.num_samples_rho)
        self.e_incz = self.compute_incident_wave()
        self.cn = self.calculate_coefficients_cn()
        self.e_scz = self.compute_scattered_field()

        #self.e_inczz_cartesian, self.e_scz_cartesian, self.e_tot = self.cartesian_conversion()

    def compute_incident_wave(self):
        e_inc_z = np.zeros((len(self.rho), len(self.phi)), dtype=complex)
        for rho_index, rho in enumerate(self.rho):
            for phi_index, phi in enumerate(self.phi):
                inc_sum: complex = 0.0
                for n in range(-int(self.num_samples_n/2), int(self.num_samples_n/2)):
                    inc_sum += 1j ** (-n) * jv(n, self.k * rho) * np.exp(1j * n * phi)
                e_inc_z[rho_index, phi_index] = self.e_inc0 * inc_sum
        return e_inc_z

    def calculate_coefficients_cn(self):
        cn = np.zeros(len(self.phi), dtype=complex)
        for n in range(-int(self.num_samples_n/2), int(self.num_samples_n/2)):
            index = n + int(self.num_samples_n/2)
            cn[index] = - (self.e_inc0 * 1j ** (-index) * jv(index, self.k * self.a)) / (hankel2(index, self.k * self.a))
        return cn

    def compute_scattered_field(self):
        e_scz = np.zeros((len(self.rho), len(self.phi)), dtype=complex)
        for rho_index, rho in enumerate(self.rho):
            for phi_index, phi in enumerate(self.phi):
                sc_sum: complex = 0.0
                for n in range(-int(self.num_samples_n/2), int(self.num_samples_n/2)):
                    n_index = n + int(self.num_samples_n/2)
                    sc_sum = self.cn[n_index] * hankel2(n, self.k * rho) * np.exp(1j * n * phi)
                e_scz[rho_index, phi_index] = sc_sum
        return e_scz

    def cartesian_conversion(self, xlim, ylim):
        x_vals = np.linspace(-xlim, xlim, self.num_samples_rho)
        y_vals = np.linspace(-ylim, ylim, self.num_samples_rho)
        #sc_interpolator = RegularGridInterpolator((self.rho, self.phi), self.)

    #def plot_incident_field(self, ax1: plt.Axes, ax2: plt.Axes):


if __name__ == '__main__':
    cylinder_scatterer = PEC_CYLINDER_TM(1.0, 1.0)
    R, Phi = np.meshgrid(cylinder_scatterer.rho, cylinder_scatterer.phi)
    # 3) Flatten for Scatterpolar
    r_flat = R.ravel()
    theta_flat = np.degrees(Phi.ravel())  # Plotly uses degrees by default
    mag_flat = abs(cylinder_scatterer.e_scz).ravel()

    # 4) Create polar scatter plot with color = |E|
    fig = go.Figure(
        go.Scatterpolar(
            r=r_flat,
            theta=theta_flat,
            mode='markers',
            marker=dict(
                size=4,
                color=mag_flat,
                colorscale='Viridis',
                colorbar=dict(title='|Eₛ𝚌,z|'),
                showscale=True,
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Scattered Field Magnitude |Eₛ𝚌,z(ρ, φ)| on Polar Grid",
        polar=dict(
            radialaxis=dict(title="ρ"),
            angularaxis=dict(direction="counterclockwise", rotation=0),  # 0° along +x
        )
    )

    fig.show()
