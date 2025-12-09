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
from cylindrical_plane_wave import PLANE_WAVE_EXCITATION
from coordinate_transforms import cyl_to_cart


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
        self.rho = np.linspace(a, self.max_rho, self.num_samples_rho)
        self.e_incz = PLANE_WAVE_EXCITATION(
            e_0=self.e_inc0,
            k=self.k,
            num_samples=self.num_samples_rho,
            rho_max=self.max_rho,
            n=self.num_samples_n,
        )
        self.cn = self.calculate_coefficients_cn()
        print(f'cn: {self.cn}')
        self.e_scz = self.compute_scattered_field()
        print(f'scattered field: {self.e_scz}')

    def calculate_coefficients_cn(self):
        cn = np.zeros(2 * self.num_samples_n, dtype=complex)
        for n in range(-self.num_samples_n, self.num_samples_n):
            index = n + self.num_samples_n
            cn[index] = - (self.e_inc0 * 1j ** (-n) * jv(n, self.k * self.a)) / (hankel2(n, self.k * self.a))
        return cn

    def compute_scattered_field(self):
        e_scz = np.zeros((len(self.rho), len(self.phi)), dtype=complex)
        for rho_index, rho in enumerate(self.rho):
            for phi_index, phi in enumerate(self.phi):
                sc_sum: complex = 0.0
                for n in range(-self.num_samples_n, self.num_samples_n):
                    n_index = n + self.num_samples_n
                    sc_sum += self.cn[n_index] * hankel2(n, self.k * rho) * np.exp(1j * n * phi)
                    #print(f'sc_sum[{n}]: {sc_sum}')
                e_scz[rho_index, phi_index] = sc_sum
        return e_scz

    # TODO: Transfer this function to another module for generic plotting utility
    def plot_scattered_field(self, title: str = None):
        x, y, f_xy = cyl_to_cart(self.rho, self.phi, self.e_scz)
        X, Y = np.meshgrid(x, y)

        plt.contourf(X, Y, f_xy.real)
        if title is None:
            title = f"Scattered Field Magnitude"
        else:
            title = title
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    cylinder_scatterer = PEC_CYLINDER_TM(1.0, (2 * np.pi / 1.0))
    cylinder_scatterer.plot_scattered_field()
    """R, Phi = np.meshgrid(cylinder_scatterer.rho, cylinder_scatterer.phi)
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

    fig.show()"""
