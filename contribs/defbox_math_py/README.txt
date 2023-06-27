#!/bin/bash

python generate_lengths_angles_dat_file.py
python deformation_math_xstampv2.py lengths_angles_input.dat
