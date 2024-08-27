import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    # Create major grid
    n1_major = 6
    n2_major = 11

    # Create minor grid (horizontal)
    n1_minor = 51
    n2_minor = 101

    fac = 100

    plt.figure()

    # Draw grid lines
    for i in range(n1_major):
        dz = 10.0
        x1 = [0.0, 100.0]
        z1 = [i * dz , i * dz]
        plt.plot(-0.5 + np.array(x1) / fac, np.array(z1) / fac, 'k.-', markersize=1.0)

    for i in range(n2_major):
        dx = 10.0
        x1 = [i * dx , i * dx]
        z1 = [0.0 , 50.0]
        plt.plot(-0.5 + np.array(x1) / fac, np.array(z1) / fac, 'k.-', markersize=1.0)

    for i in range(n1_minor):
        dz = 1.0
        x1 = [0.0, 100.0]
        z1 = [i * dz , i * dz]
        plt.plot(-0.5 + np.array(x1) / fac, np.array(z1) / fac, 'k.-', markersize=1.0, linewidth=0.2)

    for i in range(n2_minor):
        dx = 1.0
        x1 = [i * dx , i * dx]
        z1 = [0.0 , 50.0]
        plt.plot(-0.5 + np.array(x1) / fac, np.array(z1) / fac, 'k.-', markersize=1.0, linewidth=0.2)

    # Draw red dots
    for i in range(0, n2_minor, 10):
        dz = 1.0
        x1 = [50.0]
        z1 = [i * dz]
        plt.plot(-0.5 + np.array(x1) / fac, np.array(z1) / fac, 'ro', markersize=5.0)

    plt.axis([-.5, .5, 0, .5])
    plt.gca().set_aspect('equal')
    plt.xlabel(r"x")
    plt.ylabel(r"z")

    savefig_fname = "Lippmann-Schwinger/Fig/p03_grid_plot.pdf"
    plt.savefig(savefig_fname, format="pdf", bbox_inches="tight", pad_inches=0.01)

    plt.show()
