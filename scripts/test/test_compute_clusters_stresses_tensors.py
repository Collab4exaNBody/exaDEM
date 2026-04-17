import subprocess
import numpy as np
import os
from lib.io_utils import assert_files_close

# ----------------------------
# Utils
# ----------------------------
def normalize_line(line):
    parts = line.split()
    if parts and parts[0].lower() == "shapefile":
        parts[1] = os.path.basename(parts[1])  # garde juste out.shp
    return " ".join(parts)

def read_file_lines(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def compare_shp(file1, file2, tol=1e-6):
    lines1 = read_file_lines(file1)
    lines2 = read_file_lines(file2)

    assert len(lines1) == len(lines2), "Nombre de lignes différent"

    for i, (l1, l2) in enumerate(zip(lines1, lines2)):

        # Essayer de parser en float
        try:
            nums1 = list(map(float, l1.split()))
            nums2 = list(map(float, l2.split()))

            assert np.allclose(nums1, nums2, atol=tol), (
                f"Ligne {i} différente (num):\n{nums1}\n{nums2}"
            )

        except ValueError:
            # fallback texte si non numérique
            assert l1 == l2, f"Ligne {i} différente (texte):\n{l1}\n{l2}"


# ----------------------------
# Test principal
# ----------------------------

def test_compute_clusters_stresses_tensors_regression(tmp_path):
    """
    Non-regression test for the compute_clusters_stresses_tensors script. It runs the script on a simple input configuration and compares the generated .txt files to reference files. .

    """

    input_folder = os.path.abspath("test/data/compute_clusters_stresses_tensors/")    
    ref_wall_forces = os.path.abspath("test/data/compute_clusters_stresses_tensors/wall_forces.txt")
    ref_stress_tensors = os.path.abspath("test/data/compute_clusters_stresses_tensors/clusters_stresses_tensor.txt")

    out_wall_forces = tmp_path / "wall_forces.txt"
    out_stress_tensors = tmp_path / "clusters_stresses_tensor.txt"

    # --- run script ---
    script_path = os.path.abspath("post_processing/compute_clusters_stresses_tensors.py")

    cmd = [
      "python3",
      script_path,
      "--dir",
      input_folder
    ]

    result = subprocess.run(
      cmd,
      cwd=tmp_path,
      capture_output=True,
      text=True
    )
    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)

    assert result.returncode == 0

    # --- check files exist ---
    assert os.path.exists(out_wall_forces)
    assert os.path.exists(out_stress_tensors)

    #  --- compare files ---
    assert_files_close(out_wall_forces, ref_wall_forces)
    assert_files_close(out_stress_tensors, ref_stress_tensors)

