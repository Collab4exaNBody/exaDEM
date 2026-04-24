import sys
import os

from lib.io_utils import read_rockable_file, write_rockable_file
from lib.data_class import RockableData


# =========================
# FLOAT COMPARISON
# =========================
def almost_equal(a, b, tol=1e-12):
    return abs(a - b) < tol


# =========================
# STRUCTURE COMPARISON
# =========================
def compare_data(d1: RockableData, d2: RockableData):
    errors = []

    # --- params ---
    for key in d1.params.values:   
        if key not in d2.params.values:
            errors.append(f"Param missing: {key}")
            continue

        v1 = d1.params.values[key]
        v2 = d2.params.values[key]

        if isinstance(v1, list):
            for i, (a, b) in enumerate(zip(v1, v2)):
                if isinstance(a, float):
                    if not almost_equal(a, float(b)):
                        errors.append(f"{key}[{i}] diff: {a} vs {b}")
                else:
                    if a != b:
                        errors.append(f"{key}[{i}] diff: {a} vs {b}")
        else:
            if isinstance(v1, float):
                if not almost_equal(v1, float(v2)):
                    errors.append(f"{key} diff: {v1} vs {v2}")
            else:
                if v1 != v2:
                    errors.append(f"{key} diff: {v1} vs {v2}")

    # --- interactions ---
    for name, table in d1.interactions.parameters.tables.items():
        if name not in d2.interactions.parameters.tables:
            errors.append(f"Interaction missing: {name}")
            continue

        for pair in d1.interactions.parameters.tables[name]:
            v1 = d1.interactions.parameters.tables[name][pair]
            v2 = d2.interactions.parameters.tables[name].get(pair)

            if v2 is None:
                errors.append(f"{name}{pair} missing")
            elif not almost_equal(v1, v2):
                errors.append(f"{name}{pair} diff: {v1} vs {v2}")

    # --- particles ---
    if len(d1.particles) != len(d2.particles):
        errors.append("Number of particles differs")

    for i, (p1, p2) in enumerate(zip(d1.particles, d2.particles)):
        #champs scalaires
        for attr in ["name", "group", "cluster", "homothety"]:
            v1 = getattr(p1, attr)
            v2 = getattr(p2, attr)
            if v1 != v2:
                errors.append(f"Particle {i} {attr} diff: {v1} vs {v2}")    
        
        #vecteurs
        for attr in ["pos", "vel", "acc", "quat", "vrot", "arot"]:
            v1 = getattr(p1, attr)
            v2 = getattr(p2, attr)
            for j, (a, b) in enumerate(zip(v1, v2)):
                if not almost_equal(a, b):
                    errors.append(f"Particle {i} {attr}[{j}] diff: {a} vs {b}")

    # --- stick distance ---
    if d1.stick_distance is None and d2.stick_distance is not None:
        pass
    elif d1.stick_distance is not None and d2.stick_distance is None:
        errors.append("stick_distance presence diff")
    elif not almost_equal(d1.stick_distance, d2.stick_distance):
        errors.append(f"stick_distance diff: {d1.stick_distance} vs {d2.stick_distance}")   

    return errors


# =========================
# TEST PYTHON IO ROCKABLE
# =========================
def test_io_rockable_roundtrip(tmp_path):
    '''
    Test the round-trip of reading and writing a RockableData object to ensure that the IO functions are consistent. It reads a reference .txt file, writes it back, and then reads the generated file again to compare the two data structures. Any differences in the parameters, interactions, particles, or stick_distance will be reported as errors. The test will pass if no differences are found.   
    '''
    input_file = "test/data/io_rockable/input.conf"
    assert os.path.exists(input_file), f"Missing file: {input_file}"

    output_file = tmp_path / "out.txt"

    data1 = read_rockable_file(input_file)
    write_rockable_file(output_file, data1)
    data2 = read_rockable_file(output_file)

    errors = compare_data(data1, data2)

    assert errors == [], "Differences found:\n" + "\n".join(errors[:50])
