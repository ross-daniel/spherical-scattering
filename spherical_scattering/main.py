"""
All code written by Ross Stauder at Colorado State University
rev1: Dec 2025
Ross.Stauder@colostate.edu

[1] J. Jin, Theory and Computation of Electromagnetic Fields, 2nd ed. Hoboken, NJ, USA: Wiley-IEEE Press, 2015.
"""


from matplotlib import pyplot as plt

from spherical_scattering import (
    pec_sphere, pec_cylinder, dielectric_cylinder, cylindrical_plane_wave, spherical_plane_wave
)

def main():
    # First, plot the approximation of a plane wave in cylindrical coordinates for a multitude of terms
    # [1] Fig. 6.7
    cylindrical_plane_wave.main()

    # Next, use this plane wave representation as the excitation to a PEC cylinder for both TE and TM cases
    # [1] Fig. 6.9 / 6.10
    pec_cylinder.main()

    # Now, do the same for a dielectric cylinder (TM mode only)
    # [1] Fig. 6.12 (a),(c),(e)
    dielectric_cylinder.main()

    # Now, represent the plane wave in spherical coordinates
    # [1] Fig. 7.4
    spherical_plane_wave.main()

    # Then, use the spherical representation of the plane wave to excite a PEC sphere
    # [1] Fig. 7.6 and 7.8
    pec_sphere.main()

if __name__ == "__main__":
    main()

