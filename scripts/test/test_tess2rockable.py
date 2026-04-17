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


def try_parse_mixed_line(line):
    tokens = line.split()
    floats = []
    non_floats = []

    for t in tokens:
        try:
            floats.append(float(t))
            non_floats.append(None)
        except ValueError:
            non_floats.append(t)

    return floats, non_floats


def compare_shp(file1, file2, tol=1e-6):
    lines1 = read_file_lines(file1)
    lines2 = read_file_lines(file2)

    assert len(lines1) == len(lines2), "Nombre de lignes différent"

    for i, (l1, l2) in enumerate(zip(lines1, lines2)):

        tokens1 = l1.split()
        tokens2 = l2.split()

        assert len(tokens1) == len(tokens2), (
            f"Ligne {i} longueur différente:\n{l1}\n{l2}"
        )

        for t1, t2 in zip(tokens1, tokens2):
            try:
                f1 = float(t1)
                f2 = float(t2)

                assert np.isclose(f1, f2, atol=tol), (
                    f"Ligne {i} float différent:\n{f1} vs {f2}"
                )

            except ValueError:
                assert t1 == t2, (
                    f"Ligne {i} texte différent:\n{t1} vs {t2}"
                )

def compare_conf(file1, file2, tol=1e-6):
    lines1 = [normalize_line(l) for l in read_file_lines(file1)]
    lines2 = read_file_lines(file2)

    assert len(lines1) == len(lines2), "Nombre de lignes différent"

    for i, (l1, l2) in enumerate(zip(lines1, lines2)):
        tokens1 = l1.split()
        tokens2 = l2.split()

        assert len(tokens1) == len(tokens2), (
            f"Ligne {i} longueur différente:\n{l1}\n{l2}"
        )

        for t1, t2 in zip(tokens1, tokens2):
            try:
                f1 = float(t1)
                f2 = float(t2)

                assert np.isclose(f1, f2, atol=tol), (
                    f"Ligne {i} float différent:\n{f1} vs {f2}"
                )

            except ValueError:
                assert t1 == t2, (
                    f"Ligne {i} texte différent:\n{t1} vs {t2}"
                )

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

    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)

    assert result.returncode == 0

    # --- check files exist ---
    assert os.path.exists(out_conf)
    assert os.path.exists(out_shp)

    # --- compare SHP (tolérance float) ---
    compare_shp(out_shp, ref_shp)
    compare_conf(out_conf, ref_conf)
