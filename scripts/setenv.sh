#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/lib:$PYTHONPATH"

