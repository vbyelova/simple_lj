"""A Python code to simulate the unfolding of a three-bead model using the Bell equation.
Particles experience Lennard-Jones interactions with each other and bond via harmonic springs.

Made by Victoria Byelova under the supervision of Dr David Head and Prof. Lorna Dougan."""

import os, sys, pygame, math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from numpy import random


dt = 0.01                                       # timestep
num_par = 0                                   # number of particles
boxlength = 30                                # length of one side of box
energy_barrier = 10
time = 5                                        # simulation time

eq_time = time                                     # equilibration time
mass = 1                                        # mass of a particle. should be unitary
radius = 1                                      # this is our lengthscale
l_0 = 2 * radius                                # equilibrium bond length
sigma = 2.5                                     # cutoff distance for interactions. keep this pretty small
epsilon = 5                                     # potential well, around 5, in units of kT
k_bond = 2                                     # spring constant for bond

image = pygame.image.load("redsphere.png")

class Particle():
    info = """A sphere that has kinetic energy and can experience Lennard-Jones 
    interactions with other particles. If multiple particles are made, the velocity distribution is
    normalised. The particle coordinates are also randomised. Each particle has an image and rect for
    pygame."""

    def __init__(self):                                             # add vx, vy, x, y when testing
        """initialise"""
        self.vx = random.normal(loc=0, scale=0.75, size=(1, 1))     # normally distributed velocity
        self.vy = random.normal(loc=0, scale=0.75, size=(1, 1))
        self.x = random.uniform(-0.5 * boxlength, 0.5 * boxlength)  # random coordinates in box
        self.y = random.uniform(-0.5 * boxlength, 0.5 * boxlength)
#        self.vx = vx                                               # use for testing
#        self.vy = vy
#        self.x = x
#        self.y = y
        self.bonded_particles = defaultdict(list)                   # dictionary for indexing bonds
        self.image = image                                          # image object to visualise
        self.rect = image.get_rect(center = (self.x, self.y))       # assigns a space for visualisation
        self.hasbeenbroken = "no"
        self.in_three_bead = "no"
        self.polymer_id = 0
    
    def ke(self):
        """Returns kinetic energy of particle."""
        return 0.5 * mass * (self.vx**2 + self.vy**2)
    
    def update(self,coords, row_num, column_num):
        """A coordinate updating function for use in pygame. Stored coordinates from the simulation are
        passed back to the particle so the positions can be updated in real time."""
        self.x = coords[row_num, column_num]
        self.y = coords[row_num, column_num + 1]


def three_bead():
    p1, p2, p3 = Particle(), Particle(), Particle()
    p2.x, p2.y = 0, 0
    ideal_sep = l_0 - 0.02 * l_0
    p1.x = p2.x - 0.5 * ideal_sep
    p1.y = p2.y + np.sqrt(3)/2 * ideal_sep
    p3.x = p2.x + 0.5 * ideal_sep
    p3.y = p2.y + np.sqrt(3)/2 * ideal_sep
    unique_number = random.uniform(0,1)
    for p in (p1,p2,p3):
        p.in_three_bead = "yes"
        p.polymer_id = unique_number
    
    return p1, p2, p3

def add_three_bead(particles, num_par):
    polymer = three_bead()
    b0, b1, b2 =  polymer[0], polymer[1], polymer[2]
    particles.append(b0)
    particles.append(b1)
    particles.append(b2)
    num_par += 3

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
    Boundary check is performed to find the shortest interaction distance. All values are converted to floats 
    for consistency and removal of TypeErrors. """

    rx = float(j.x - i.x)
    ry = float(j.y - i.y)
    rx, ry = boundary_check(rx, ry)
    r2 = float(rx**2 + ry**2)
    rij = float(np.sqrt(r2))
    return rx, ry, r2, rij


def boundary_check(rx, ry):
    """Checks if a particle's coordinates are outside the boundary conditions.
    If so, the position is corrected. In some cases, the boundary conditions are used to wrap the interaction 
    around the box instead of through. In other instances, the particle coordinates are changed."""
    while rx >= 0.5 * boxlength:
        rx -= boxlength
    while rx <= -0.5 * boxlength:
        rx += boxlength
    while ry >= 0.5 * boxlength:
        ry -= boxlength
    while ry <= -0.5 * boxlength:
        ry += boxlength

    return rx, ry


def lj_force(i, j, distance_store):
    """Finds the separation between two particles, checks if the particles are
    within box boundaries and finds shortest separation. If separation is above the
    cut-off, no force is returned. Otherwise, the force due to LJ interaction
    is returned for the x-direction and the y-direction."""
    #print("lj force")
    rx, ry, r2 = distance_store[0], distance_store[1], distance_store[2]

    if r2 > sigma**2:
        return [0, 0]
    rij = distance_store[3]
    #print(rij)

    vec_sep = [rx, ry]

    rhat = vec_sep / (np.abs(rij))

    if r2 <= sigma**2:
        magnitude = (48 * epsilon * (sigma**-1)
            * (((sigma / rij) ** 13) - 0.5 * ((sigma / rij) ** 7)))
        return magnitude * rhat


def lj_energy(i, j, distance_store):
    """Returns the Lennard-Jones energy due to short-range particle interactions."""
    #print("lj energy")
    rij = distance_store[3]
    return 4 * epsilon * (((sigma / rij) ** 12 - ((sigma / rij) ** 6)))


def make_step(i, j, t, bonds, distance_store):
    """Calculates the LJ force ands incorporates into Euler method to find new
    velocities and coordinates. If the simulation is still within the equilibration time, a force ceiling is
     created to prevent explosions.
    """
    #print(i.vx, i.x, dt, "sanity check")

    lj = lj_force(i, j, distance_store)
    #print("lj force:", lj)
    #print(i.vx, i.x, dt, "sanity check after")
    b = bond_force(i, j, bonds, distance_store)
    f = np.add(lj, b)
    #print("old force: ", f)
    if t < eq_time:
        #print("time working", t)
        if np.linalg.norm(f) > 200:
            #print("bond force", b)
            ##print("lj force", lj)
            f = (f / (np.linalg.norm(f))) * 150
            #print("I EXPLODED AT T = ", t)
            #print(f, t)
            #print("coordinates", i.x , i.y, j.x, j.y)
        #    print("working")
        #elif f[1].any() >= 200 or f[1].any() <= -200:
        #    f[1] = (f[1] / (np.abs(f[1]))) * 150
    #print("corrected force: ", f[0], f[1])
    if [i, j] not in bonds:
        if (i.hasbeenbroken and j.hasbeenbroken == "yes"
             and i.polymer_id == j.polymer_id):
            bm = bell_model(i, j, f)
            f = np.subtract(f, bm)

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


def stick(i, j, distance_store):
    """Checks if the particle separation is within the LJ potential well. Returns 1 if
        particles are close enough to bond and 0 if their separation is too large."""
    if i.polymer_id == j.polymer_id and i.hasbeenbroken == "yes" and j.hasbeenbroken == "yes":
        return 0
    rij = distance_store[3]
    if np.abs(rij) >= l_0:
        return 0
    elif np.abs(rij) <  l_0:
        return 1


def bond_energy(i, j, distance_store):
    """Calculates the harmonic bond energy of two particles."""
    #print("bond energy")
    rij = distance_store[3]
    return 0.5 * k_bond * (rij - l_0) ** 2


def bond_force(i, j, bonds, distance_store):
    """Checks the bond list to see if the two particles are bonded. If not, a force of 0 is returned.
        If a bond does exist between the two particles, the force in x- and y-directions is returned."""
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
    
def bell_model(i, j, f):
    attempt_freq = 1
    k_bT = 1
    transition_dx = 0.01
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

def visualise(particles, coords):
    """Uses pygame library to visualise particle movement over course of simulation. Coordinates and bonded pair are 
    extracted from simulation and drawn in pygame window."""
    pygame.init()
    clock = pygame.time.Clock()                                         # creates object to track time
    offset =  63                                                        # shifts bonds to right position
    screen = pygame.display.set_mode((boxlength * 15, boxlength * 15))  # sets up a visualisation space
    modified_coords = (coords * 0.2 * boxlength) + 10 * boxlength       # makes a new array of shifted coordinates
    running =  True
    row_num = 0                                                         # equivalent to timestep
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False                                         # means we can close the pygame window manually
        screen.fill((255,255,255))
        column_num = 0
        if row_num == modified_coords.shape[0]:                         # stops the visualisation if we run out of coords
            running = False
        else:
            for p in particles:
                
                p.update(modified_coords, row_num, column_num)          # updates particle positions
                screen.blit(image,(p.x, p.y))                           # draws particles
                if row_num not in p.bonded_particles.keys():            # skips bond drawing if there are none during t
                    pass
                else:
                    for val in p.bonded_particles[row_num]:             # draws lines for all bonds during t
                        rx = p.x - modified_coords[row_num, val * 2]
                        ry = p.y - modified_coords[row_num, val *2 + 1]
                        boundary_lim = (0.2 * boxlength + 10 * boxlength) * 0.5
                        if (rx >= boundary_lim or rx <= -1 * boundary_lim
                        or ry >= boundary_lim  or ry <= -1 * boundary_lim):
                            pass
                        else:
                            pygame.draw.line(screen, (0, 0, 0),
                            (modified_coords[row_num, val * 2] + offset , 
                            modified_coords[row_num, val * 2 + 1] + offset ),
                            (p.x + offset, p.y + offset))
                column_num += 2
            row_num += 1
        pygame.display.flip()                                           # updates all of the screen
        clock.tick(60)                                                  # visualises at 30 fps




def simulate():
    """Generates particles and calculates their new position according to the
    force experienced. Plots the coordinates."""
    particles = []                                                      # particles will be made and added to this list
    t = 0                                                               # time counter
    bonds = []
    particles = [Particle() for _ in range(num_par)]                    # makes a list of particles
#    add_three_bead(particles, num_par)
    add_three_bead(particles, num_par)
#    particles.append(Particle(-5,0,5,0))                               # use for testing
#    particles.append(Particle(5,0,-5,0))
    coords = np.zeros((int(time/dt) + 1, 2 * len(particles)))           # array for storing coordinates
    row_num = 0                                                         # each row in coords represents a timestep
    while t < time:
        column_num = 0
        for num, p1 in enumerate(particles):
            for p2 in particles[num + 1 :]:
                distance_store = distance_calc(p1, p2)                  # calculate separation of particles
                #print(particles[0].x, "before")    
                make_step(p1, p2, t, bonds, distance_store)             # calculates forces, modifies velocities and position
                #print(new_bond)
                if [p1,p2] not in bonds:
                    new_bond = stick(p1, p2, distance_store)                        # tests if particles are close enough to bond
                    if new_bond == 1:
                        bonds.append([p1, p2])                          # keep track of how many bonds exists and which ones
                if [p1,p2] in bonds:
                    p1.bonded_particles[row_num].append(particles.index(p2))
                    if p1.polymer_id == p2.polymer_id and p1.hasbeenbroken == "no" and p2.hasbeenbroken == "no":
                        total_energy = np.add(bond_energy(p1,p2,distance_store),lj_energy(p1,p2,distance_store))
                        if total_energy < energy_barrier:
                            pass
                        if total_energy >= energy_barrier:
                            print("let's break pls")
                            bonds.remove([p1,p2])
                            p1.hasbeenbroken = "yes"
                            p2.hasbeenbroken = "yes"
            coords[row_num, column_num] = p1.x                          # each particle has two columns. one for x, one for y
            coords[row_num, column_num+1] = p1.y
            column_num += 2                                             # coordinates of this particle have been saved. move on to next
        for p in particles:
            boundary_check(p.x, p.y)                                    # put any stray particles back in the box
        t += dt
        row_num += 1                                                    # updates in sync with t, means each move at each t is saved
    print(len(bonds), "bonds")  
    visualise(particles, coords)
    return len(bonds)

simulate()
#file_check()
# os.system("ffmpeg -f image2 -r 5 -i ./plots/graph_%d.png ./video/test1.mp4")
