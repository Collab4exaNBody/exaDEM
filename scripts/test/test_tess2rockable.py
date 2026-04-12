import subprocess
import numpy as np
import os


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

def test_tess2rockable_regression(tmp_path):
    """
    Non-regression test for the tess2rockable script. It runs the script on a simple input tessellation and compares the generated .conf and .shp files to reference files. The .shp comparison is done with a tolerance for floating-point values, while the .conf comparison is done as a strict text comparison (you can improve this by parsing the .conf files if needed).

    """

    input_tess = "test/data/tess2rockable/cube.tess"
    ref_conf = "test/data/tess2rockable/out_sticked.conf"
    ref_shp = "test/data/tess2rockable/out.shp"

    out_conf = tmp_path / "out_sticked.conf"
    out_shp = tmp_path / "out.shp"

    # --- run script ---
    cmd = [
        "python3",
        "pre_processing/tess2rockable.py",
        input_tess,
        "1.e-4",
         str(out_shp),
    ]
    print("Running command:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)


    # Debug utile si ça casse
    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)

    assert result.returncode == 0

    # --- check files exist ---
    assert os.path.exists(out_conf)
    assert os.path.exists(out_shp)

    # --- compare SHP (tolérance float) ---
    compare_shp(out_shp, ref_shp)

    conf_new = [normalize_line(l) for l in read_file_lines(out_conf)]
    conf_ref = read_file_lines(ref_conf)

    assert conf_new == conf_ref, "wrong .conf content"