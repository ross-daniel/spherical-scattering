import numpy as np
import matplotlib.pyplot as plt
from scipy.special import (
    jv, jvp, h2vp, hankel2
)
import pandas as pd
from cylindrical_plane_wave import (
    PLANE_WAVE_EXCITATION_TM, PLANE_WAVE_EXCITATION_TE
)
from coordinate_transforms import cyl_to_cart


class PEC_CYLINDER_TM:
    def __init__(self, e_incident0: complex, k: float, a: float = 1.0, num_samples_n: int = 40,
                 num_samples: int = 100, max_rho: float = 5.0 * np.sqrt(2)):
        self.e_inc0 = e_incident0
        self.num_samples_n = num_samples_n
        self.k = k
        self.a = a
        self.num_samples = num_samples
        self.max_rho = max_rho
        self.e_incz = PLANE_WAVE_EXCITATION_TM(
            e_0=self.e_inc0,
            k=self.k,
            num_samples=self.num_samples,
            rho_max=self.max_rho,
            n=self.num_samples_n,
        )
        self.x = self.e_incz.x
        self.y = self.e_incz.y
        self.xy_pairs = self.e_incz.xy_pairs
        self.rho_phi_pairs = self.e_incz.rho_phi_pairs
        self.cn = self.calculate_coefficients_cn()
        #print(f'cn: {self.cn}')
        self.e_scz = self.compute_scattered_field()
        #print(f'scattered field: {self.e_scz}')
        self.e_tot = self.compute_total_field()

    def calculate_coefficients_cn(self):
        cn = np.zeros(2 * self.num_samples_n, dtype=complex)
        for n in range(-self.num_samples_n, self.num_samples_n):
            index = n + self.num_samples_n
            cn[index] = - (self.e_inc0 * 1j ** (-n) * jv(n, self.k * self.a)) / (hankel2(n, self.k * self.a))
        return cn

    def compute_scattered_field(self):
        e_scz = np.zeros(len(self.rho_phi_pairs), dtype=complex)
        for index, (rho, phi) in enumerate(self.rho_phi_pairs):
            sc_sum: complex = 0.0
            if rho <= self.a:
                e_scz[index] = 0.0
            else:
                for n in range(-self.num_samples_n, self.num_samples_n):
                    n_index = n + self.num_samples_n
                    sc_sum += self.cn[n_index] * hankel2(n, self.k * rho) * np.exp(1j * n * phi)
                    #print(f'sc_sum[{n}]: {sc_sum}')
                e_scz[index] = sc_sum
        return e_scz

    def compute_total_field(self):
        return self.e_incz.e_inc_z_cylindrical + self.e_scz

    # TODO: Transfer this function to another module for generic plotting utility
    def plot_scattered_field(self, title: str = None):
        f_xy = np.copy(self.e_scz)
        #for index, (rho, phi) in enumerate(self.rho_phi_pairs):
            #if rho <= self.a:
                #f_xy[index] = 0.0

        f_xy = np.reshape(f_xy, (len(self.x), len(self.y)))
        #x, y, f_xy = cyl_to_cart(self.rho, self.phi, self.xy_pairs)
        X, Y = np.meshgrid(self.x, self.y)
        fig, ax = plt.subplots()

        cs = ax.contourf(X, Y, f_xy.real)
        if title is None:
            title = f"Real Part of Scattered Electric Field"
        else:
            title = title
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(cs, ax=ax)
        # circle
        circ = plt.Circle((0, 0), self.a, fill=True, color='gray')
        ax.add_patch(circ)
        plt.show()

    def plot_total_field(self, title: str = None):
        #for index, (rho, phi) in enumerate(self.rho_phi_pairs):
            #if rho <= self.a:
                #f_xy[index] = 0.0

        f_xy = np.reshape(self.e_tot, (len(self.x), len(self.y)))
        #x, y, f_xy = cyl_to_cart(self.rho, self.phi, self.xy_pairs)
        X, Y = np.meshgrid(self.x, self.y)

        fig, ax = plt.subplots()

        cs = ax.contourf(X, Y, f_xy.real)
        if title is None:
            title = f"Real Part of Total Electric Field"
        else:
            title = title
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(cs, ax=ax)
        # circle
        circ = plt.Circle((0, 0), self.a, fill=True, color='gray')
        ax.add_patch(circ)
        plt.show()


class PEC_CYLINDER_TE:
    def __init__(self, h_incident0: complex, k: float, a: float = 1.0, num_samples_n: int = 40,
                 num_samples: int = 100, max_rho: float = 5.0 * np.sqrt(2)):
        self.h_inc0 = h_incident0
        self.num_samples_n = num_samples_n
        self.k = k
        self.a = a
        self.num_samples = num_samples
        self.max_rho = max_rho
        self.h_incz = PLANE_WAVE_EXCITATION_TE(
            h_0=self.h_inc0,
            k=self.k,
            num_samples=self.num_samples,
            rho_max=self.max_rho,
            n=self.num_samples_n,
        )
        self.x = self.h_incz.x
        self.y = self.h_incz.y
        self.xy_pairs = self.h_incz.xy_pairs
        self.rho_phi_pairs = self.h_incz.rho_phi_pairs
        self.bn = self.calculate_coefficients_bn()
        # print(f'cn: {self.cn}')
        self.h_scz = self.compute_scattered_field()
        # print(f'scattered field: {self.e_scz}')
        self.h_tot = self.compute_total_field()

    def calculate_coefficients_bn(self):
        bn = np.zeros(2 * self.num_samples_n, dtype=complex)
        for n in range(-self.num_samples_n, self.num_samples_n):
            n_index = n + self.num_samples_n
            bn[n_index] = - (1j ** (-n)) * jvp(n, self.k * self.a, 1) / h2vp(n, self.k * self.a, 1)
        return bn

    def compute_scattered_field(self):
        h_scz = np.zeros(len(self.rho_phi_pairs), dtype=complex)
        for index, (rho, phi) in enumerate(self.rho_phi_pairs):
            sc_sum: complex = 0.0
            if rho <= self.a:
                h_scz[index] = 0.0
            else:
                for n in range(-self.num_samples_n, self.num_samples_n):
                    n_index = n + self.num_samples_n
                    sc_sum += self.bn[n_index] * hankel2(n, self.k * rho) * np.exp(1j * n * phi)
                h_scz[index] = sc_sum
        return h_scz

    def compute_total_field(self):
        return self.h_scz + self.h_incz.h_inc_z_cylindrical

    def plot_scattered_field(self, title: str = None):
        f_xy = np.copy(self.h_scz)

        f_xy = np.reshape(f_xy, (len(self.x), len(self.y)))
        #x, y, f_xy = cyl_to_cart(self.rho, self.phi, self.xy_pairs)
        X, Y = np.meshgrid(self.x, self.y)
        fig, ax = plt.subplots()

        cs = ax.contourf(X, Y, f_xy.real)
        if title is None:
            title = f"Real Part of Scattered Magnetic Field"
        else:
            title = title
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(cs, ax=ax)
        # circle
        circ = plt.Circle((0, 0), self.a, fill=True, color='gray')
        ax.add_patch(circ)
        plt.show()

    def plot_total_field(self, title: str = None):
        f_xy = np.reshape(self.h_tot, (len(self.x), len(self.y)))
        #x, y, f_xy = cyl_to_cart(self.rho, self.phi, self.xy_pairs)
        X, Y = np.meshgrid(self.x, self.y)

        fig, ax = plt.subplots()

        cs = ax.contourf(X, Y, f_xy.real)
        if title is None:
            title = f"Real Part of Total Magnetic Field"
        else:
            title = title
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(cs, ax=ax)
        # circle
        circ = plt.Circle((0, 0), self.a, fill=True, color='gray')
        ax.add_patch(circ)
        plt.show()


if __name__ == '__main__':
    # TM Mode
    #pec_cylinder_tm = PEC_CYLINDER_TM(1.0, (2 * np.pi / 1.0))
    #pec_cylinder_tm.plot_scattered_field()
    #pec_cylinder_tm.plot_total_field()

    # TE Mode
    pec_cylinder_te = PEC_CYLINDER_TE(1.0, (2 * np.pi / 1.0))
    pec_cylinder_te.plot_scattered_field()
    pec_cylinder_te.plot_total_field()

