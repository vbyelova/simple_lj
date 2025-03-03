"""A Python code to simulate the trajectories of particles in a box. The particles experience short-range
lj interactions.

Made by Victoria Byelova with the supervision of Dr David Head and Prof. Lorna Dougan."""

import os, sys, pygame, math
import matplotlib.pyplot as plt
import numpy as np
from numpy import random


dt = 0.01  # timestep
num_par = 10  # number of particles
boxlength = 30

eq_time = 0.1
time = 2
mass = 1
radius = 1  # this is our lengthscale
l_0 = 2 * radius  # equilibrium bond length
sigma = 2  # cutoff distance for interactions. keep this pretty small
epsilon = 5  # around 5, units of kT
k_bond = 50

image = pygame.image.load("redsphere.png")

class Particle():
    info = """A sphere that has kinetic energy and can experience Lennard-Jones 
    interactions with other particles. If multiple particles are made, the velocity distribution is
    normalised. The particle coordinates are also randomised. Each particle has an image and rect for
    pygame."""

    def __init__(self):  # velocity components, coordinates, radius
        """initialise"""
        self.vx = random.normal(loc=0, scale=0.75, size=(1, 1))
        self.vy = random.normal(loc=0, scale=0.75, size=(1, 1))
        self.x = random.uniform(-0.5 * boxlength, 0.5 * boxlength)
        self.y = random.uniform(-0.5 * boxlength, 0.5 * boxlength)
#        self.vx = vx
#        self.vy = vy
#        self.x = x
#        self.y = y
        self.image = image

        self.rect = image.get_rect(center = (self.x, self.y))

    def ke(self):
        """Returns kinetic energy of particle."""
        return 0.5 * mass * (self.vx**2 + self.vy**2)
    
    def update(self,coords, row_num, column_num):
        """A coordinate updating function for use in pygame. Stored coordinates from the simulation are
        passed back to the particle so the positions can be updated in real time."""
        self.x = (coords[row_num, column_num] + 0.3 * boxlength) * 10
        self.y = (coords[row_num, column_num + 1] + 0.3 * boxlength) * 10




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
    Boundary check is performed to find the shortest interaction distance.
    """

    rx = float(j.x - i.x)
    ry = float(j.y - i.y)
    boundary_check(rx, ry)
    r2 = float(rx**2 + ry**2)
    rij = float(np.sqrt(r2))
    return rx, ry, r2, rij


def boundary_check(rx, ry):
    """Checks if a particle's coordinates are outside the boundary conditions.
    If so, the position is corrected. In some cases, the boundary conditions are used to wrap the interaction 
    around the box instead of through. In other instances, the particle coordinates are changed."""
    if rx >= 0.5 * boxlength:
        rx -= boxlength
    elif rx <= -0.5 * boxlength:
        rx += boxlength
    elif ry >= 0.5 * boxlength:
        ry -= boxlength
    elif ry <= -0.5 * boxlength:
        ry += boxlength

    return rx, ry


def lj_force(i, j, distance_store):
    """Finds the separation between two particles, checks if the particles are
    within PBD and finds shortest separation. If separation is above the
    cut-off, no force is returned. Otherwise, the force due to LJ interaction
    is returned for the x-direction and the y-direction."""
    #print("lj force")
    rx, ry, r2 = distance_store[0], distance_store[1], distance_store[2]
#    i.x, i.y = boundary_check(i.x, i.y, distance_store)
#    j.x, j.y = boundary_check(j.x, j.y, distance_store)
#    rx, ry = boundary_check(rx, ry, distance_store)
    
    if r2 > sigma**2:
        return [0, 0]
    rij = distance_store[3]
    #print(rij)
    vec_sep = [rx, ry]


    if rij > 0.5 * boxlength:
        rij = boxlength - rij
    elif rij < -0.5 * boxlength:
        rij = boxlength + rij
    rhat = vec_sep / (np.abs(rij))

    if r2 <= sigma**2:
        magnitude = (
            48
            * epsilon
            * (sigma**-1)
            * (((sigma / rij) ** 13) - 0.5 * ((sigma / rij) ** 7))
        )
        return magnitude * rhat


def lj_energy(i, j, distance_store):
    """Returns the Lennard-Jones energy due to short-range particle interactions."""
    #print("lj energy")
    rij = distance_store[3]
    return 4 * epsilon * (((sigma / rij) ** 12 - ((sigma / rij) ** 6)))


def make_step(i, j, t, bonds, distance_store):
    """Calculates the LJ force ands incorporates into Euler method to find new
    velocities and coordinates. If the simulation is still in the equilibration time, a force ceiling is
     created to prevent explosions.
    """
    #print(i.vx, i.x, dt, "sanity check")

    lj = lj_force(i, j, distance_store)
    print("lj force:", lj)
    #print(i.vx, i.x, dt, "sanity check after")
    b = bond_force(i, j, bonds, distance_store)
    f = np.add(lj, b)
    #print("old force: ", f)
    if t < eq_time:
        #print("time working", t)
        if np.linalg.norm(f) > 200:
            f = (f / (np.linalg.norm(f))) * 150
        #    print(f, t)
        #    print("working")
        #elif f[1].any() >= 200 or f[1].any() <= -200:
        #    f[1] = (f[1] / (np.abs(f[1]))) * 150
    #print("corrected force: ", f[0], f[1])
    i.vx = i.vx - f[0] * dt
    i.x = i.x + i.vx * dt
    i.vy = i.vy - f[1] * dt
    i.y = i.y + i.vy * dt

    j.vx = j.vx + f[0] * dt
    j.x = j.x + j.vx * dt
    j.vy = j.vy + f[1] * dt
    j.y = j.y + j.vy * dt
    return


def graph(beads, t):
    """Plots a graph and saves a snapshot to a folder to then be made into an mp4."""

    plt.axis([-0.5 * boxlength, 0.5 * boxlength, -0.5 * boxlength, 0.5 * boxlength])
    for bead in beads:
        plt.plot(bead.x, bead.y, marker=".")
    plt.savefig("./three_bead_plots/graph_%d.png" % t)

    return


def stick(distance_store):
    """Checks if the particle separation is within the LJ potential well. Returns """
#    i.x, i.y = boundary_check(i.x, i.y, distance_store)
#    j.x, j.y = boundary_check(j.x, j.y, distance_store)
#    rx, ry = boundary_check(rx, ry, distance_store)
    rij = distance_store[3]
    if np.abs(rij) > l_0:
        return 0
    elif np.abs(rij) <= l_0:
        return 1


def bond_energy(i, j, distance_store):
    """Calculates the energy of a bond between two particles."""
    #print("bond energy")
    rij = distance_store[3]
    return 0.5 * k_bond * (rij - l_0) ** 2


def bond_force(i, j, bonds, distance_store):
    """Checks if a bond between two particles exists on the bond list. If not, there is no bond force experienced.
    If so, the particle experiences a spring force from the bond."""
    #print("bond force")
    rx, ry = distance_store[0], distance_store[1]
    if [i, j] not in bonds:
        return [0, 0]
    else:
        rij = distance_store[3]
        vec_sep = [rx, ry]
        rhat = vec_sep / (np.abs(rij))
        magnitude = -1 * k_bond * (rij - l_0)
        return magnitude * rhat

def visualise(particles, coords, bond_indices):
    """Uses pygame library to visualise particle movement over course of simulation. Coordinates and bonded pair are 
    extracted from simulation and drawn in pygame window."""
    pygame.init()
    clock = pygame.time.Clock()
    offset = 0.35 * boxlength
    screen = pygame.display.set_mode((boxlength * 20, boxlength * 20))
    running =  True
    row_num = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((255,255,255))
        column_num = 0
        if row_num == coords.shape[0]:
            running = False
        else:
            for p in particles:
                p.update(coords, row_num, column_num)
                screen.blit(image,(p.x, p.y))
                column_num += 2
            row_num += 1
        pygame.display.flip()
        clock.tick(30)



def simulate():
    """Generates particles and calculates their new position according to the
    force experienced. Plots the coordinates."""
    particles = []  # Particles will be made and added to this list.
    t = 0  # time counter
    bonds = []
    bond_indices = []
    particles = [Particle() for _ in range(num_par)]  # makes a list of particles
#    particles.append(Particle(-0.5,0,1,0))
#    particles.append(Particle(0.5,0,-1,0))
    coords = np.zeros((int(time/dt) + 1, 2 * len(particles)))
    row_num = 0
    while t < time:
        column_num = 0
        for num, p1 in enumerate(particles):
            for p2 in particles[num + 1 :]:
                distance_store = distance_calc(p1, p2)
                #print(particles[0].x, "before")    
                make_step(p1, p2, t, bonds, distance_store)
                new_bond = stick(distance_store)
                #print(new_bond)
                if new_bond == 1:
                    if [p1,p2] not in bonds:
                        bonds.append([p1, p2])
                        bond_indices.append([row_num, p1, p2])
            coords[row_num, column_num] = p1.x
            coords[row_num, column_num+1] = p1.y
            column_num += 2
        for p in particles:
            boundary_check(p.x, p.y)
        t += dt
        row_num += 1
    print(coords)
    visualise(particles, coords, bonds)
    return print(len(bonds), "bonds")#, plt.show()


#file_check()
simulate()
# os.system("ffmpeg -f image2 -r 5 -i ./plots/graph_%d.png ./video/test1.mp4")
