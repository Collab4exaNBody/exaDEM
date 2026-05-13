#!/usr/bin/env python3

import numpy as np
import glob
import os
import re
import argparse


from collections import defaultdict

from lib.io_utils import read_interactions

def read_tracked_contacts(filename):
    tracked = {}

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # ignorer lignes vides ou commentaires
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Ligne invalide dans {filename}: {line}")

            i, j, si, sj = map(int, parts[:4])
            name = parts[4]

            key = (min(i, j), max(i, j), min(si, sj), max(si, sj))
            tracked[key] = name

    return tracked

# ==============================
# MAIN
# ==============================

def main():
    
    parser = argparse.ArgumentParser(
    description="Plot interface statistics from simulation files",
    epilog= """
    Usage : python plot_interfaces_stat.py --dir /path/to/simulation/files --tstep 1.0
    This script reads the interaction files from a simulation, extracts the normal and tangential forces for specific contacts, and saves the time series of these forces in a text file for further analysis or plotting.
    Input:
    - Interaction files in the format InteractionOutputDir-*/InteractionOutputDir-*_0.txt
    Output:
    - debug_contacts.txt: a text file containing the time series of normal and tangential forces for the tracked contacts.
    Example:
    python plot_interfaces_stat.py --dir /path/to/simulation/files --tstep 1.0
    """   
    )
    
    parser.add_argument("--dir", type=str, default=".",
                    help="Directory containing simulation files (default=current directory)")
    parser.add_argument("--tracked", type=str, required=True,
                    help="File containing tracked contacts")
    parser.add_argument("--tstep", type=float, default=1.0,
                        help="Time step (default=1.0)")
    args = parser.parse_args()

    tstep = args.tstep

    # INTERACTION FILES
    # -------------------------
    base_dir = args.dir
    interaction_files = glob.glob(
        os.path.join(base_dir, "InteractionOutputDir-*", "InteractionOutputDir-*_0.txt")
    )

    # sort interaction files by step number
    interaction_files = sorted(
        interaction_files,
        key=lambda x: int(re.findall(r"InteractionOutputDir-(\d+)", x)[-1])
    )   
    # tracked contacts: (i, j, si, sj) -> name

    tracked = read_tracked_contacts(args.tracked)


    # Exemple de fichier de contacts à suivre (format : i j si sj name) :
    # i j si sj name
    # 0 1 1 1 c1
    # 0 1 3 3 c2
    # 0 1 5 5 c3
    # 0 1 7 7 c4

    # tracked = {
    #     (0,1,1,1): "c1",
    #     (0,1,3,3): "c2",
    #     (0,1,5,5): "c3",
    #     (0,1,7,7): "c4",
    # }

    data = {
        name: {"t": [], "fnx": [], "fny": [], "fnz": [], "ftx": [], "fty": [], "ftz": []}
        for name in tracked.values()
    }

    for inter_file in interaction_files:

        step = int(re.findall(r"InteractionOutputDir-(\d+)", inter_file)[-1])
        time = step * tstep

        contacts = read_interactions(inter_file)

        for c in contacts:

            quadruple = (min(c.i, c.j), max(c.i, c.j), min(c.si, c.sj), max(c.si, c.sj))

            if quadruple not in tracked:
                continue

            name = tracked[quadruple]

            if not (c.type <= 3 or c.type == 13):
                continue

            data[name]["t"].append(time)
            data[name]["fnx"].append(c.fn[0])
            data[name]["fny"].append(c.fn[1])
            data[name]["fnz"].append(c.fn[2])
            data[name]["ftx"].append(c.ft[0])
            data[name]["fty"].append(c.ft[1])
            data[name]["ftz"].append(c.ft[2])

    with open("debug_contacts.txt", "w") as f:

        header = "time"
        for name in data:
            header += f" {name}_fnx {name}_fny {name}_fnz {name}_ftx {name}_fty {name}_ftz"
        f.write(header + "\n")

        n = min(len(data[name]["t"]) for name in data)#on ne garde que les temps communs à tous les contacts pour éviter les problèmes d'alignement des données, au cas où certains contacts n'existent pas à tous les pas de temps.
        # n = len(data["c1"]["t"])

        for i in range(n):

            line = f"{data['c1']['t'][i]}"

            for name in data:
                line += f" {data[name]['fnx'][i]} {data[name]['fny'][i]} {data[name]['fnz'][i]} {data[name]['ftx'][i]} {data[name]['fty'][i]} {data[name]['ftz'][i]}"

            f.write(line + "\n")


if __name__ == "__main__":
    main()
