#!/usr/bin/env python3

import numpy as np
import glob
import os
import re
import argparse


from collections import defaultdict

from lib.io_utils import read_xyzdem_snapshot
from lib.data_utils import build_particle_index, build_clusters


# ==============================
# Physics
# ==============================

def compute_delta(particles):
    '''
    Compute the vertical extent of the particle assembly (delta_z) and the maximum and minimum z coordinates.
    
    Parameters
    ----------
    particles : list of Particle
        list of Particle objects representing the particles in the system
        
    Returns
    ----------
    delta_x : float
        the horizontal extent of the particle assembly in the x direction (xmax - xmin)
    delta_y : float
        the horizontal extent of the particle assembly in the y direction (ymax - ymin)
    delta_z : float
        the vertical extent of the particle assembly (zmax - zmin)  
    
    '''   
    x = [p.pos[0] for p in particles]
    y = [p.pos[1] for p in particles]
    z = [p.pos[2] for p in particles]


    return max(x) - min(x), max(y) - min(y), max(z) - min(z)


def compute_wall_force(contacts):
    '''
    Compute the total vertical force exerted by the particles on the top and bottom walls.

    Parameters
    ----------
    contacts : list of Contact
        list of Contact objects representing the contacts in the system
    
    Returns
    ----------
    Fx_top : float
        total horizontal force exerted on the top wall (positive if to the right)   
    Fx_bottom : float
        total horizontal force exerted on the bottom wall (positive if to the right)
    Fy_top : float
        total horizontal force exerted on the top wall (positive if upwards)
    Fy_bottom : float
        total horizontal force exerted on the bottom wall (positive if upwards)
    Fz_top : float
        total vertical force exerted on the top wall (positive if upwards)
    Fz_bottom : float
        total vertical force exerted on the bottom wall (positive if downwards)
    '''
    contacts_x = []
    contacts_y = []
    contacts_z = []


    for c in contacts:
        if 4 <= c.type <= 12:
            fx = c.force[0]
            fy = c.force[1]
            fz = c.force[2]
            xc = 0.5 * (c.pos_i[0] + c.pos_j[0])
            yc = 0.5 * (c.pos_i[1] + c.pos_j[1])
            zc = 0.5 * (c.pos_i[2] + c.pos_j[2])

            contacts_x.append((xc, fx))
            contacts_y.append((yc, fy))
            contacts_z.append((zc, fz))

    if not contacts_x:
        Fx_top = 0.0
        Fx_bottom = 0.0
    else:
        x_vals = [x for x, _ in contacts_x]
        x_mid = 0.5 * (max(x_vals) + min(x_vals))
        Fx_top = sum(fx for x, fx in contacts_x if x > x_mid)
        Fx_bottom = sum(-fx for x, fx in contacts_x if x <= x_mid)
    
    if not contacts_y:
        Fy_top = 0.0
        Fy_bottom = 0.0
    else:
        y_vals = [y for y, _ in contacts_y]
        y_mid = 0.5 * (max(y_vals) + min(y_vals))
        Fy_top = sum(fy for y, fy in contacts_y if y > y_mid)
        Fy_bottom = sum(-fy for y, fy in contacts_y if y <= y_mid)
    
    if not contacts_z:
        Fz_top = 0.0
        Fz_bottom = 0.0
    else:
        z_vals = [z for z, _ in contacts_z]
        z_mid = 0.5 * (max(z_vals) + min(z_vals))
        Fz_top = sum(fz for z, fz in contacts_z if z > z_mid)
        Fz_bottom = sum(-fz for z, fz in contacts_z if z <= z_mid)
        

    return Fx_top,Fx_bottom, Fy_top, Fy_bottom, Fz_top, Fz_bottom


def compute_stress_tensor(cluster_particles, contacts, particle_index,density=1.0):
    '''
    Compute the stress tensor for a given cluster of particles using the contact forces and positions.

    Parameters
    ----------
    cluster_particles : list of Particle
        list of Particle objects belonging to the cluster for which to compute the stress tensor
    contacts : list of Contact
        list of Contact objects representing the contacts in the system
    particle_index : dict
        dictionary mapping particle IDs to Particle objects

    Returns
    ----------
    sigma : numpy.ndarray
        the stress tensor for the given cluster of particles
    '''
    sigma = np.zeros((3, 3))

    if not cluster_particles:
        return sigma
    
    if all(p.mass is not None for p in cluster_particles):
      V = sum(p.mass / density for p in cluster_particles)
    else:
      V = len(cluster_particles)

    cluster_ids = {p.id for p in cluster_particles}

    for c in contacts:
        i, j = c.i, c.j

        if c.type == 13:
            continue

        in_i = i in cluster_ids
        in_j = j in cluster_ids

        if not (in_i or in_j):
            continue

        # -------------------------
        # WALL CONTACT
        # -------------------------
        if 4 <= c.type <= 12:
            if in_i:
                f = c.force
                x = np.array(c.pos_i)
            elif in_j:
                f = -np.array(c.force)
                x = np.array(c.pos_j)
            else:
                continue

        # -------------------------
        # PARTICLE CONTACT
        # -------------------------
        elif 0 <= c.type <= 3:

            if in_i and in_j:
                # même cluster → ignore
                if particle_index[i].cluster == particle_index[j].cluster:
                    continue

            if in_i:
                f = c.force
                x = np.array(c.pos_i)
            elif in_j:
                f = -np.array(c.force)
                x = np.array(c.pos_j)
            else:
                continue

        else:
            continue

        contact_point = 0.5 * (np.array(c.pos_i) + np.array(c.pos_j))
        sigma += np.outer(f, contact_point)

    return sigma / V


# ==============================
# MAIN
# ==============================

def main():
    
    parser = argparse.ArgumentParser(
    description="Compute stress tensors and wall forces from DEM simulation files (.xyz + interaction files).",
    epilog= """
    Usage: python compute_clusters_stresses_tensors.py --density 2500
    Input:
    - .xyz files containing particle positions and velocities (e.g., dem_pos_vel_*.xyz)
    - Corresponding interaction files in InteractionOutputDir-*/InteractionOutputDir-*_0.txt

    Output:
    - clusters_stresses_tensor.txt: time series of stress tensor components for each cluster
    - wall_forces.txt: time series of wall forces and vertical extent of the particle assembly

    Example:
    python compute_clusters_stresses_tensors.py --density 2500
    """   
    )
    
    parser.add_argument("--dir", type=str, default=".",
                    help="Directory containing simulation files (default=current directory)")
    parser.add_argument("--density", type=float, default=1.0,
                        help="Particle density (default=1.0)")
    args = parser.parse_args()

    density = args.density

    # FILES
    # -------------------------
    base_dir = args.dir
    particle_files = sorted(glob.glob(os.path.join(base_dir, "*.xyz")))
   
    # -------------------------
    # OUTPUT FILES
    # -------------------------
    f_stress = open("clusters_stresses_tensor.txt", "w")
    f_wall = open("wall_forces.txt", "w")

    # headers
    f_stress.write("#time id_cluster sxx syy szz sxy sxz syx syz szx szy\n")
    f_wall.write("#time delta_x delta_y delta_z esp_x esp_y esp_z Fx Fy Fz\n")

    forces = []
    deltas = []
    L0 = None

    for pf in particle_files:

        step = str(int(re.findall(r"(\d+)", pf)[-1]))
        inter_file = os.path.join(
          base_dir,
          f"InteractionOutputDir-{step}",
          f"InteractionOutputDir-{step}_0.txt"
     )
        if not os.path.exists(inter_file):
            continue

        data = read_xyzdem_snapshot(pf, inter_file)

        time = data.time

        particles = data.particles
        contacts = data.interactions.contacts

        particle_index = build_particle_index(particles)
        clusters = build_clusters(particles)

        # -------------------------
        # STRESS
        # -------------------------
        for cid, plist in clusters.items():

            sigma = compute_stress_tensor(
                plist,
                contacts,
                particle_index,
                density=density
            )

            f_stress.write(
                f"{time} {cid} "
                f"{sigma[0,0]} {sigma[1,1]} {sigma[2,2]} "
                f"{sigma[0,1]} {sigma[0,2]} "
                f"{sigma[1,0]} {sigma[1,2]} "
                f"{sigma[2,0]} {sigma[2,1]}\n"
            )
        # -------------------------
        # WALL FORCE
        # -------------------------
        Fx_top, Fy_top, Fz_top, Fx_bottom, Fy_bottom, Fz_bottom = compute_wall_force(contacts)

        Fx = 0.5 * (Fx_top + Fx_bottom)
        Fy = 0.5 * (Fy_top + Fy_bottom)
        Fz = 0.5 * (Fz_top + Fz_bottom)

        delta_x, delta_y, delta_z = compute_delta(particles)
        if L0 is None:
           L0 = (delta_x, delta_y, delta_z)

        Lx0, Ly0, Lz0 = L0

        eps_x = (delta_x - Lx0) / Lx0
        eps_y = (delta_y - Ly0) / Ly0
        eps_z = (delta_z - Lz0) / Lz0

        f_wall.write(
            f"{time} {delta_x} {delta_y} {delta_z} {eps_x} {eps_y} {eps_z} {Fx} {Fy} {Fz}\n"
        )

    # -------------------------
    # CLOSE FILES
    # -------------------------
    f_stress.close()
    f_wall.close()


if __name__ == "__main__":
    main()