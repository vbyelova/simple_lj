"""A Python code to simulate the trajectories of particles in a box. The particles experience short-range
vdw attraction and bond via springs. 

Made by Victoria Byelova with the supervision of Dr David Head and Prof. Lorna Dougan."""

import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import random

# matplotlib.use("Agg")


dt = 0.01  # timestep
num_par = 50  # number of particles
boxlength = 20

eq_time = 0.1
time = 10
mass = 1
radius = 0.5  # this is our lengthscale
l_0 = 2 * radius  # equilibrium bond length
sigma = 2  # cutoff distance for interactions. keep this pretty small
epsilon = 0  # around 5, units of kT
k_bond = 5
bonds = []


class Particle:
    info = """A sphere that has kinetic energy and can experience Lennard-Jones 
    interactions with other particles. If multiple particles are made, the velocity distribution is
    normalised. The particle coordinates are also randomised."""

    def __init__(self):  # velocity components, coordinates, radius
        """initialise"""
        self.vx = random.normal(loc=0, scale=0.75, size=(1, 1))
        self.vy = random.normal(loc=0, scale=0.75, size=(1, 1))
        self.x = random.uniform(-0.5 * boxlength, 0.5 * boxlength)
        self.y = random.uniform(-0.5 * boxlength, 0.5 * boxlength)

        """self.vx = vx
        self.vy = vy
        self.x = x
        self.y = y"""

    def ke(self):
        return 0.5 * mass * (self.vx**2 + self.vy**2)


def file_check():
    """Checks if directories for video/snapshot storage exist, and makes them if not."""
    if os.path.isdir("./plots/"):
        pass
    else:
        os.mkdir("./plots/")

    if os.path.isdir("./video/"):
        pass
    else:
        os.mkdir("./video/")
    return


def distance_calc(i, j):
    """A function to calculate the separation between two particles. In the main code,
    this is used to make a data array that all of the functions can access and calculations are made just once.
    """
    rx = j.x - i.x
    ry = j.y - i.y
    r2 = rx**2 + ry**2
    return [rx, ry, r2]


def boundary_check(rx, ry):
    """Checks if a particle's coordinates are outside the boundary conditions.
    If so, the position is corrected."""
    rx, ry = float(rx), float(ry)
    if rx >= 0.5 * boxlength:
        rx -= boxlength
    elif rx <= -0.5 * boxlength:
        rx += boxlength
    elif ry >= 0.5 * boxlength:
        ry -= boxlength
    elif ry <= -0.5 * boxlength:
        ry += boxlength

    return rx, ry


def lj_force(i, j):
    """Finds the separation between two particles, checks if the particles are
    within PBD and finds shortest separation. If separation is above the
    cut-off, no force is returned. Otherwise, the force due to LJ interaction
    is returned for the x-direction and the y-direction."""

    i.x, i.y = boundary_check(i.x, i.y)
    j.x, j.y = boundary_check(j.x, j.y)
    rx, ry, r2 = distance_store[0], distance_store[1], distance_store[2]
    rx, ry = boundary_check(rx, ry)
    if r2 > sigma**2:
        return np.array([0, 0])
    rij = np.sqrt(r2)
    # print(rij)
    vec_sep = np.array([rx, ry])

    if rij > 0.5 * boxlength:
        rij = boxlength - rij
    elif rij < -0.5 * boxlength:
        rij = boxlength + rij
    rhat = vec_sep / (np.abs(rij))

    if r2 <= sigma**2:
        return (
            48
            * epsilon
            * (sigma**-1)
            * (((sigma / rij) ** 13) - 0.5 * ((sigma / rij) ** 7))
            * rhat
        )


def lj_energy(i, j):
    """Returns the Lennard-Jones energy experience due to short-range particle interactions."""
    rx, ry, r2 = (
        (distance_calc(i, j))[0],
        (distance_calc(i, j))[1],
        (distance_calc(i, j))[2],
    )
    rij = np.sqrt(r2)
    return 4 * epsilon * (((sigma / rij) ** 12 - ((sigma / rij) ** 6)))


def make_step(i, j, t):
    """Calculates the LJ force ands incorporates into Euler method to find new
    velocities and coordinates. If the simulation is still in the equilibration time, a force ceiling is created to prevent explosions.
    """
    lj = lj_force(i, j)
    b = bond_force(i, j)
    f = np.add(lj, b)
    # print("old force: ", f)
    if t < eq_time:
        if f[0].any() > 200 or f[0].any() <= -200:
            f[0] = (f[0] / (np.abs(f[0]))) * 150
        elif f[1].any() >= 200 or f[1].any() <= -200:
            f[1] = (f[1] / (np.abs(f[1]))) * 150
    # print("corrected force: ", f[0], f[1])
    i.vx = i.vx - f[0] * dt
    i.x = i.x + i.vx * dt
    i.vy = i.vy - f[1] * dt
    i.y = i.y + i.vy * dt

    j.vx = j.vx + f[0] * dt
    j.x = j.x + j.vx * dt
    j.vy = j.vy + f[1] * dt
    j.y = j.y + j.vy * dt

    return


def graph(i, j, t):
    """Plots a graph and saves a snapshot to a folder to then be made into an mp4."""

    plt.axis([-0.5 * boxlength, 0.5 * boxlength, -0.5 * boxlength, 0.5 * boxlength])
    plt.plot(i.x, i.y, marker=".")
    plt.plot(j.x, j.y, marker=".")
    plt.savefig("./plots/graph_%d.png" % t)

    return


def stick(i, j):
    """Checks if particles are within the box boundaries. If so and the particles are close enough, the particles are appended to a list."""
    i.x, i.y = boundary_check(i.x, i.y)
    j.x, j.y = boundary_check(j.x, j.y)
    rx, ry, r2 = (
        (distance_calc(i, j))[0],
        (distance_calc(i, j))[1],
        (distance_calc(i, j))[2],
    )
    rx, ry = boundary_check(rx, ry)
    rij = np.sqrt(r2)
    if np.abs(rij) <= l_0:
        return [i, j]
    else:
        return


def bond_energy(i, j):
    """Calculates the energy of a bond between two particles."""
    rx, ry, r2 = (
        (distance_calc(i, j))[0],
        (distance_calc(i, j))[1],
        (distance_calc(i, j))[2],
    )
    rij = np.sqrt(r2)
    return 0.5 * k_bond * (rij - l_0) ** 2


def bond_force(i, j):
    """Checks if a bond between two particles exists on the bond list. If not, there is no bond force experienced.
    If so, the particle experiences a spring force from the bond."""
    rx, ry, r2 = (
        (distance_calc(i, j))[0],
        (distance_calc(i, j))[1],
        (distance_calc(i, j))[2],
    )
    if [i, j] not in bonds:
        return [0, 0]
    rij = np.sqrt(r2)
    vec_sep = np.array([rx, ry])
    rhat = vec_sep / (np.abs(rij))
    return -1 * k_bond * (rij - l_0) * rhat


def simulate():
    """Generates particles and calculates their new position according to the
    force experienced. Plots the coordinates."""
    particles = []  # Particles will be made and added to this list.
    data = []  # The coordinates and velocity of the particle will be added here.
    energy = []
    total_energy = []
    t = 0  # time counter
    particles = [Particle() for _ in range(num_par)]  # makes a list of particles
    # particles.append(Particle(-0.5,0,1,0))
    # particles.append(Particle(0.5,0,-1,0))

    plt.subplots(2, 1, figsize=(5, 10))
    plt.subplot(2, 1, 1)
    plt.xlabel("x position")
    plt.ylabel("y position")

    global distance_store
    while t < time:
        for num, p1 in enumerate(particles):
            for p2 in particles[num + 1 :]:

                distance_store = np.array(distance_calc(p1, p2))

                make_step(p1, p2, t)
                new_bond = stick(p1, p2)
                if isinstance(new_bond, list):
                    if new_bond not in bonds:
                        bonds.append(new_bond)
                    energy.append(bond_energy(p1, p2))

                energy.append(lj_energy(p1, p2))
                energy.append(p1.ke())
                energy.append(p2.ke())
                total_energy.append([t, sum(energy)])
                energy.clear()
                data.append([p1.x, p1.y, p2.x, p2.y])
                plt.scatter(p1.x, p1.y, marker=".")
                plt.scatter(p2.x, p2.y, marker=".")

                # graph(p1,p2,t)

                t += dt

    total_energy = np.array(total_energy, dtype="object")
    data = np.array(data)

    plt.subplot(2, 1, 2)
    plt.xlabel("time")
    plt.ylabel("energy")
    plt.plot(total_energy[:, 0], total_energy[:, 1])

    return print(len(bonds), "bonds"), plt.show()


file_check()
simulate()
# os.system("ffmpeg -f image2 -r 5 -i ./plots/graph_%d.png ./video/test1.mp4")
