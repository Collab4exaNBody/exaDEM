import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt

# ==============================
# Utils
# ==============================

def parse_time(header_line):
    match = re.search(r"Time=([0-9.eE+-]+)", header_line)
    return float(match.group(1)) if match else None


def parse_vec3(string):
    string = string.strip().replace("(", "").replace(")", "")
    return np.array([float(x) for x in string.split()])


# ==============================
# Lecture des particules
# ==============================

def read_particles(filename):
    particles = {}
    clusters = {}

    with open(filename, 'r') as f:
        n = int(f.readline())
        header = f.readline()
        time = parse_time(header)
        for line in f:
            data = line.split()
            pid = int(data[-1])
            pos = np.array(list(map(float, data[1:4])))
            vel = np.array(list(map(float, data[4:7])))
            cluster_id = int(data[7])
            mass = float(data[8])

            particles[pid] = {
                "pos": pos,
                "vel": vel,
                "cluster": cluster_id,
                "mass": mass
            }

            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(pid)

    return time, particles, clusters


# ==============================
# Lecture des interactions
# ==============================

def read_interactions(filename):
    interactions = []
    particle_ids = set()   # pour stocker les ids uniques


    with open(filename, 'r') as f:
        for line in f:
            data = line.strip().split(",")

            i = int(data[0])
            j = int(data[1])
            interaction_type = int(data[4])

            overlap = float(data[5])

            contact_pos = np.array(list(map(float, data[6:9])))
            normal_force = np.array(list(map(float, data[9:12])))
            tangential_force = np.array(list(map(float, data[12:15])))
            pos_i = np.array(list(map(float, data[15:18])))
            pos_j = np.array(list(map(float, data[18:21])))

            force = normal_force + tangential_force
            #print("Force:", force)
            interactions.append({
                "i": i,
                "j": j,
                "type": interaction_type,
                "force": force,
                "pos_i": pos_i,
                "pos_j": pos_j
            })

#    print(f"Nb particules uniques dans interactions : {len(particle_ids)}")

    return interactions

# ==============================
# Forces aux parois
# ==============================

def compute_wall_force(interactions):
    contacts = []

    for inter in interactions:
        if 4 <= inter["type"] <= 12:
        #if 0 <= inter["type"] <= 3:

            f = inter["force"]

            #print("force paroi =", f)
            if len(f) != 3:
              continue

            fz = f[2]
            print("fz= ", fz, "type=", inter["type"])            

            zc = 0.5 * (inter["pos_i"][2] + inter["pos_j"][2])
            contacts.append((zc, fz))

    if len(contacts) == 0:
        return 0.0, 0.0

    z_values = [c[0] for c in contacts]
    z_mid = 0.5 * (max(z_values) + min(z_values))

    Fz_top = 0.0
    Fz_bottom = 0.0

    for zc, fz in contacts:
        if zc > z_mid:
            Fz_top += fz
        else:
            Fz_bottom += -fz


    #print("Fz_top= ",Fz_top, " Fz_bottom= ",Fz_bottom)
    return Fz_top, Fz_bottom

# ==============================
# Déplacement axial
# ==============================

def compute_delta_z(particles):
    z_values = [p["pos"][2] for p in particles.values()]
    z_max = max(z_values)
    z_min = min(z_values)

    delta = z_max - z_min
    return delta, z_max, z_min

# ==============================
# Tenseur de contrainte
# ==============================

def compute_stress_tensor(cluster_particles, interactions, particles):
    sigma = np.zeros((3, 3))

    # Volume (penser a utiliser la densite)
    V = sum(p["mass"] for p in cluster_particles.values())

    if V == 0:
        return sigma

    cluster_ids = set(cluster_particles.keys())

    for inter in interactions:
        i, j = inter["i"], inter["j"]
        itype = inter["type"]
        
        #ignore innerbonds
        if itype == 13:
            continue
        
        in_i = i in cluster_ids
        in_j = j in cluster_ids
        
        #ignore if not belong to the current cluster
        if not (in_i or in_j):
            continue
      
            
        # =========================
        # CAS 1 : interaction mur
        # =========================
        if 4 <= itype <= 12:


            if in_i:
                f = inter["force"]
                x = inter["pos_i"]
            elif in_j:
                f = -inter["force"]
                x = inter["pos_j"]
            else:
                continue

        # =========================
        # CAS 2 : particule-particule
        # =========================
        elif 0 <= itype <= 3:

            # cluster différent
            if in_i and in_j:
                # ignore si meme cluster
                 if particles[i]["cluster"] == particles[j]["cluster"]:
                    continue

            if in_i:
                f = inter["force"]
                x = inter["pos_i"]
            elif in_j:
                f = -inter["force"]
                x = inter["pos_j"]
            else:
                continue

        else:
            continue

        contact_point = 0.5 * (inter["pos_i"] + inter["pos_j"])
        sigma += np.outer(f, contact_point)
        #sigma += np.outer(f, x)

    return sigma / V

# ==============================
# MAIN
# ==============================

def main():

    particle_files = sorted(glob.glob("dem_pos_vel_*.xyz"))
    times = []
    cluster_stress = {}
    
    forcesu = []
    forcesb = []
    deltas = []
    
    
    for pf in particle_files:
        
        # fichier interaction correspondant
        step_str = re.findall(r"(\d+)", pf)[-1]
        step = str(int(step_str))  # supprime les zéros en tête

        inter_file = f"InteractionOutputDir-{step}/InteractionOutputDir-{step}_0.txt"

        if not os.path.exists(inter_file):
            continue
        time, particles, clusters = read_particles(pf)
        interactions = read_interactions(inter_file)

        times.append(time)

        for cid, pids in clusters.items():
            cluster_particles = {pid: particles[pid] for pid in pids}

            sigma = compute_stress_tensor(cluster_particles, interactions,particles)

            if cid not in cluster_stress:
                cluster_stress[cid] = []

           # contrainte moyenne (trace)
           # trace = np.trace(sigma) / 3.0
           # cluster_stress[cid].append(trace)

           # contrainte moyenne (trace)
            sigma_zz = sigma[2, 2]
            cluster_stress[cid].append(sigma_zz)
 
        Fz_top, Fz_bottom = compute_wall_force(interactions)
        Fz = 0.5 * (Fz_top + Fz_bottom)  

        delta, zmax, zmin = compute_delta_z(particles)

        #forces.append(Fz)
        forcesu.append(Fz_top)
        forcesb.append(Fz_bottom)
        deltas.append(delta)

    # ==============================
    # Plot
    # ==============================

    plt.figure()

    for cid, values in cluster_stress.items():
        plt.plot(times[:len(values)], values, label=f"Cluster {cid}")

    plt.xlabel("Time")
    plt.ylabel("Sigzz")
    plt.legend()
    plt.grid()

    plt.show()
    
    plt.figure()
    plt.plot(times, forcesu, 'o-', label="top")
    plt.plot(times, forcesb, 's-', label="bot")
    plt.xlabel("delta")
    plt.ylabel("Force")
    plt.grid()
    plt.show()
    
    plt.figure()
    plt.plot(np.power(times, 1.5), forces, 'o')
    plt.xlabel("delta^(3/2)")
    plt.ylabel("Force")
    plt.grid()
    plt.show()
#    
#    plt.figure()
#    plt.loglog(times, forces, 'o-')
#    plt.xlabel("delta")
#    plt.ylabel("Force")
#    plt.grid()
#    plt.show()
    
    

if __name__ == "__main__":
    main()
