#!/usr/bin/env python3

import numpy as np
import glob
import os
import re
import argparse


from collections import defaultdict
from lib.io_utils import read_xyzdem_snapshot


 # Load snapshot
# ===========================================================================

def load_snapshot(filepath):
    """
    Get an ExaDEM .xyz snapshot.
    Returns (time, id_array, pos_array) with pos_array of shape (N, 3).
    """
    data = read_xyzdem_snapshot(filepath)
    time = data.time
    particles = data.particles  # liste de Particle

    ids = np.array([p.id  for p in particles], dtype=int)
    pos = np.array([p.pos for p in particles], dtype=float)  # (N, 3)
    return time, ids, pos

 # sTRAINs
# ===========================================================================
 
def compute_center_strains(ids_ref, pos_ref, ids_cur, pos_cur,
                           center, r_select, load_axis=1):
    """
    Calcule eps_load et eps_trans au centre du disque.
 
    Déplacement = pos_cur - pos_ref  (appariement par id).
 
    Parameters
    ----------
    ids_ref, pos_ref : référence (t=0)
    ids_cur, pos_cur : snapshot courant
    center           : array [cx, cy, cz]
    r_select         : rayon de sélection
 
    Returns
    -------
    eps_x, eps_y, eps_z, n_used
    """
 
    # Appariement par id 
    ref_idx = {pid: i for i, pid in enumerate(ids_ref)}
    common_ids = [pid for pid in ids_cur if pid in ref_idx]
 
    if len(common_ids) == 0:
        return np.nan, np.nan, np.nan, 0
 
    cur_idx = {pid: i for i, pid in enumerate(ids_cur)}
 
    i_ref = np.array([ref_idx[pid]   for pid in common_ids])
    i_cur = np.array([cur_idx[pid]   for pid in common_ids])
 
    pos_r = pos_ref[i_ref]
    pos_c = pos_cur[i_cur]
    disp  = pos_c - pos_r  # déplacement (N, 3)
 
    # Near center
    rel = pos_r - center
    dist = np.linalg.norm(rel, axis=1)
    mask = dist < r_select
 
    if mask.sum() < 4:
        return np.nan, np.nan, np.nan , 0
 
    rel_sel  = rel[mask]
    disp_sel = disp[mask]
 
    #Compute strain components
    def strain(grp_plus, grp_minus, disp_axis, pos_axis):

        if grp_plus.sum() == 0 or grp_minus.sum() == 0:
            return np.nan

        u_plus = disp_sel[grp_plus, disp_axis].mean()
        u_minus = disp_sel[grp_minus, disp_axis].mean()

        x_plus = rel_sel[grp_plus, pos_axis].mean()
        x_minus = rel_sel[grp_minus, pos_axis].mean()

        dx = x_plus - x_minus

        if abs(dx) < 1e-15:
            return np.nan

        return (u_plus - u_minus) / dx
 
    # eps_xx
    right = rel_sel[:, 0] > 0
    left  = rel_sel[:, 0] < 0
    eps_x = strain(right, left, 0, 0)

    # eps_yy
    top = rel_sel[:, 1] > 0
    bottom = rel_sel[:, 1] < 0
    eps_y = strain(top, bottom, 1, 1)    

    # eps_zz
    front = rel_sel[:, 2] > 0
    back = rel_sel[:, 2] < 0
    eps_z = strain(front, back, 2, 2)   


    return eps_x, eps_y, eps_z, int(mask.sum())
 

# ==============================
# MAIN
# ==============================

def main():
    
    parser = argparse.ArgumentParser(
    description="Compute stress tensors and wall forces from DEM simulation files (.xyz + interaction files).",
    epilog= """
    Usage: python compute_box_deformation.py --dir ./ --r_select 0.1 --center 0.5 0.5 0.5
    Input:
    - .xyz files containing particle positions and velocities (e.g., dem_pos_vel_*.xyz)
    - radius of selection (r_select) around the specified center
    - center coordinates (x, y, z) of the cube to analyze
        
    Output:
    - cube_deformation.txt: time series of deformation tensor components for the cube

    Example:
    python compute_box_deformation.py --dir ./ --r_select 0.1 --center 0.5 0.5 0.5
    """   
    )
    
    parser.add_argument("--dir", type=str, default=".",
                    help="Directory containing simulation files (default=current directory)")
    parser.add_argument("--r_select",  type=float, default=None,
                        help="Rayon de sélection autour du centre [m] (défaut: R/10)")
    parser.add_argument("--center",    type=float, nargs=3, default=[0., 0., 0.],
                        help="Centre du disque x y z (défaut: 0 0 0)")
    
    args = parser.parse_args()
    r_select  = args.r_select 
    center    = np.array(args.center)

    # FILE
    # -------------------------
    base_dir = args.dir
    particle_files = sorted(glob.glob(os.path.join(base_dir, "*.xyz")))

    # REFERENCE
    time_ref, ids_ref, pos_ref = load_snapshot(particle_files[0])
    print(f"Reference : {particle_files[0]}  (t={time_ref:.4e}, N={len(ids_ref)})")
   
    # -------------------------
    # OUTPUT FILES
    # -------------------------
    f_eps = open("cube_deformation.txt", "w")

    # headers
    f_eps.write("#time eps_xx eps_yy eps_zz \n")

    for pf in particle_files[1:]: #skip first file (reference)

        try:
            time, ids_cur, pos_cur = load_snapshot(pf)
        except Exception as e:
            print(f"  Error {pf} : {e}")
            continue

        eps_x, eps_y, eps_z,n_used = compute_center_strains(
            ids_ref, pos_ref, ids_cur, pos_cur,
            center, r_select
        )

        f_eps.write(f"{time:.6e} {eps_x:.6e} {eps_y:.6e} {eps_z:.6e} {n_used}\n")
        

    # -------------------------
    # CLOSE FILES
    # -------------------------
    f_eps.close()



if __name__ == "__main__":
    main()