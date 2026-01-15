#!/usr/bin/env python3
import math
import sys

def compute_damp_rate(en2):
    """
    Compute the dimensionless damping rate from the restitution coefficient.

    Parameter
    ----------
    en2 : float
        Square of the restitution coefficient.

    Returns
    -------
    damp_rate : float
        Dimensionless damping rate.
    """

    if 0.0 < en2 < 1.0:
        log_en = 0.5 * math.log(en2)
        damp_rate = -log_en / math.sqrt(log_en * log_en + math.pi * math.pi)
    elif en2 <= 0.0:
        damp_rate = 1.0
    else:
        damp_rate = 0.0

    return damp_rate


def parse_key_value_args(args):
    """
    Parse command-line arguments of the form key=value.

    Parameters
    ----------
    args : list of str

    Returns
    -------
    params : dict
        Dictionary of parsed parameters.
    """
    params = {}
    for arg in args:
        if "=" not in arg:
            raise ValueError(f"Invalid argument format: '{arg}', expected key=value")
        key, value = arg.split("=", 1)
        params[key] = float(value)
    return params


if __name__ == "__main__":
    # Parse command-line arguments
    params = parse_key_value_args(sys.argv[1:])

    # Required parameters
    en2 = params.get("en2")

    if en2 is None:
        raise ValueError("'en2' must be provided (e.g. en2=0.01)")

    damp_rate = compute_damp_rate(en2)

    print(f"damp_rate = {damp_rate:.16e}")
