"""A coarse-grained model of a polymer, composing of three beads with a temporary bond.
Upon this bond breaking, the polymer will be able to unfold. 

Written by Victoria Byelova under the supervision of Dr. David Head and Prof. Lorna Dougan."""

import os
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

#sim parameters
boxlength = 10
dt = 0.01
eq_time = 0.5
time = 10
bonds = []
energy_barrier = 0.1

#bead parameters
num_beads = 3
mass = 1
r = 1

#spring parameters
l_0 = 2 * r
k_bond = 15

#lj parameters
sigma = 3 #cutoff distance for interactions
epsilon = 1 #units of kT

class Bead:
    info = """A bead representing part of a polymer. Arbitrary values are
            set for velocities and coordinates so that they may be amended
            in other functions."""
    hasnotbeenbroken = 1
    bondindex = "empty"
    def __init__(self):
        self.mass = mass
        self.r = r
        self.x = 0
        self.y = 0
        self.vx = 0.1#random.normal(loc=0, scale=0.75, size=(1, 1))
        self.vy = 0.1#random.normal(loc=0, scale=0.75, size=(1, 1))
    
    def ke(self):
        """returns kinetic energy of the bead."""
        return 0.5 * mass * (self.vx ** 2 + self.vy ** 2)


def file_check():
    """Checks if directories for video/snapshot storage exist, and makes them if not."""
    if os.path.isdir("./three_bead_plots/"):
        pass
    else:
        os.mkdir("./three_bead_plots/")

    if os.path.isdir("./three_bead_video/"):
        pass
    else:
        os.mkdir("./three_bead_video/")
    return

def graph(i, j, t):
    """Plots a graph and saves a snapshot to a folder to then be made into an mp4."""

    plt.axis([-0.5 * boxlength, 0.5 * boxlength, -0.5 * boxlength, 0.5 * boxlength])
    plt.plot(i.x, i.y, marker=".")
    plt.plot(j.x, j.y, marker=".")
    plt.savefig("./three_bead_plots/graph_%d.png" % t)

    return

def distance_calc(i, j):
    #taken from working particles
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


def lj_energy(i, j):
    """Returns the Lennard-Jones energy experience due to short-range particle interactions."""
    r2 = (
        (distance_calc(i, j))[2],
    )
    rij = np.sqrt(r2)
    return 4 * epsilon * (((sigma / rij) ** 12 - ((sigma / rij) ** 6)))

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

def stick(i, j):
    if i.hasnotbeenbroken == 0 and j.hasnotbeenbroken == 0:
        return
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
    r2 = (
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

def make_step(i, j, t):
    #print("calculating lj")
    lj = lj_force(i, j)
    #print("calculating bond force")
    sb = bond_force(i, j)
    f = np.add(lj, sb)
    if t < eq_time:
            #print("time working", t)
            if np.linalg.norm(f) > 200:
                f = (f / (np.linalg.norm(f))) * 150
    if [i, j] not in bonds:
        #print("checking bond whilst calculating force")
        if i.hasnotbeenbroken and j.hasnotbeenbroken == 0:
            #print("adding bell model force")
            bm = bell_model(i, j, f)
            f = np.add(f, bm)
    else:

        i.vx = i.vx - f[0] * dt
        i.x = i.x + i.vx * dt
        i.vy = i.vy - f[1] * dt
        i.y = i.y + i.vy * dt

        j.vx = j.vx + f[0] * dt
        j.x = j.x + j.vx * dt
        j.vy = j.vy + f[1] * dt
        j.y = j.y + j.vy * dt
        return


def bell_model(i, j, f):
    attempt_freq = 0.7
    k_bT = 1
    transition_dx = 1
    numerator = -1 * (energy_barrier - f * transition_dx)
    denom = k_bT
    exponent = numerator / denom
    rate_constant = attempt_freq ** (exponent)
    rx, ry, r2 = (
        (distance_calc(i, j))[0],
        (distance_calc(i, j))[1],
        (distance_calc(i, j))[2],
    )
    rij = np.sqrt(r2)
    vec_sep = np.array([rx, ry])
    rhat = vec_sep / (np.abs(rij))
    return rate_constant * dt * rhat
    


def simulate():
    global distance_store
    data = []
    t = 0

    #print("making beads")
    beads = [Bead() for _ in range(num_beads)]
    beads[0].x, beads[0].y = 0, 0
    beads[1].x, beads[1].y = 1, 1
    beads[2].x, beads[2].y = -1, 1
    b3 = beads[2]

    while t < time:
        for num, b1 in enumerate(beads):
            for b2 in beads[num + 1 :]:
                distance_store = np.array(distance_calc(b1, b2))
                #print("making step")
                make_step(b1, b2, t)
                #print("checking new bond")
                new_bond = stick(b1, b2)
                if isinstance(new_bond, list):
                    if new_bond not in bonds:
                        bonds.append(new_bond)
                b_e = (bond_energy(b1, b2))
                l_e = lj_energy(b1, b2)
                if (l_e + b_e) > energy_barrier:
                    if ([b1, b2]) in bonds:
                        if([b1, b2]) == ([beads[1], beads[2]]):
                            bonds.remove([b1, b2])
                            b1.hasnotbeenbroken = 0
                            b2.hasnotbeenbroken = 0

                data.append([b1.x, b1.y, b2.x, b2.y])
                #plt.scatter(b1.x, b1.y, marker = ".")
                #plt.scatter(b2.x, b2.y, marker = ".")

                graph(b1, b2, t)
        plt.clf()       
        t += dt

    print(len(bonds), "bonds")
    #traj_fig.show()                  

#file_check()
simulate()
# os.system("ffmpeg -f image2 -r 5 -i ./three_bead_plots/graph_%d.png ./three_bead_video/test1.mp4")
