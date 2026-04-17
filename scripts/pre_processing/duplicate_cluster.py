import argparse
import copy
from io_rockable_utils import read_rockable_file, write_rockable_file

def duplicate_sphere(data, nx, ny, nz, spacing):
    """
    Duplication d'une sphère fragmentée dans un motif 3D.
    nx, ny, nz : nombre de duplications dans chaque direction
    spacing : distance entre centres des sphères
    """
    original_particles = data["particles"]
    n_particles_per_sphere = len(original_particles)
    new_particles = []
    cluster_id = 0  # on renumérote les clusters

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                offset = [ix*spacing, iy*spacing, iz*spacing]

                # Incrémentation du cluster
                cluster_id += 1

                for p in original_particles:
                    new_p = copy.deepcopy(p)
                    # Décaler la position
                    new_p["pos"] = [p["pos"][i] + offset[i] for i in range(3)]
                    # Nouveau cluster
                    new_p["cluster"] = cluster_id
                    new_particles.append(new_p)

    # Mettre à jour les particules et n_particles
    data["particles"] = new_particles
    data["n_particles"] = len(new_particles)
    return data


def main():
    parser = argparse.ArgumentParser(description="Dupliquer une sphère fragmentée Rockable")
    parser.add_argument("input_file", help="Fichier Rockable source")
    parser.add_argument("output_file", help="Fichier Rockable de sortie")
    parser.add_argument("--nx", type=int, default=1, help="Nombre de duplications en x")
    parser.add_argument("--ny", type=int, default=1, help="Nombre de duplications en y")
    parser.add_argument("--nz", type=int, default=1, help="Nombre de duplications en z")
    parser.add_argument("--spacing", type=float, default=1.0, help="Distance entre sphères")
    args = parser.parse_args()

    data = read_rockable_file(args.input_file)
    data = duplicate_sphere(data, args.nx, args.ny, args.nz, args.spacing)
    write_rockable_file(args.output_file, data)
    print(f"{data['n_particles']} particules écrites dans {args.output_file}")


if __name__ == "__main__":
    main()
