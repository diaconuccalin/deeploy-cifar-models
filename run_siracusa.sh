#!/bin/bash
# Usage: run_siracusa.sh <test dir relative to DeeployTest or absolute under /tallinn> [extra runner args...]
# Runs Deeploy codegen + GVSoC simulation for Siracusa (tiled) inside the Deeploy devel container.
set -e
TEST=$1; shift
docker run --rm -v $HOME/Deeploy:/app/Deeploy -v $HOME/tallinn:/tallinn -w /app/Deeploy \
  ghcr.io/pulp-platform/deeploy:devel bash -c "
  pip install -q -e . --no-deps > /tmp/pip.log 2>&1 || { echo 'DEEPLOY INSTALL FAILED'; cat /tmp/pip.log; exit 1; }
  cd DeeployTest && python3 deeployRunner_tiled_siracusa.py -t $TEST $*"
