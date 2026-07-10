#!/bin/bash
set -e
cd /Users/kunalbhatia/Desktop/Research/microlensing/binml/code
R2=/private/tmp/claude-501/-Users-kunalbhatia-Desktop-Research-microlensing/27fcbf76-7077-4777-82da-d78e9a275394/scratchpad/r2/shards
gen(){ python3 simulate.py --num_workers 10 --n_flat "${8:-0}" --n_pspl "${7:-0}" --n_binary "$1" \
        --binary_preset "$2" --q_min "$3" --q_max "$4" $5 --seed "$6" --output "$R2/$9"; }
echo "=== c1 gap realistic $(date +%T) ==="; gen 12000 baseline 0.01 0.1 ""               300 0 0 r2_c1_gap_real.h5
echo "=== c2 gap forced   $(date +%T) ==="; gen 6000  baseline 0.01 0.1 "--force_caustic" 301 0 0 r2_c2_gap_forced.h5
echo "=== c3 mid realistic $(date +%T) ==="; gen 10000 baseline 0.001 0.01 ""              302 0 0 r2_c3_mid_real.h5
echo "=== c4 mid forced   $(date +%T) ==="; gen 4000  baseline 0.001 0.01 "--force_caustic" 303 0 0 r2_c4_mid_forced.h5
echo "=== c5 low realistic $(date +%T) ==="; gen 14000 baseline 0.0001 0.001 ""             304 0 0 r2_c5_low_real.h5
echo "=== c6 low forced   $(date +%T) ==="; gen 4000  baseline 0.0001 0.001 "--force_caustic" 305 0 0 r2_c6_low_forced.h5
echo "=== c7 stellar      $(date +%T) ==="; gen 8000  stellar  0.1 1 ""                     306 0 0 r2_c7_stellar.h5
echo "=== c8 fullrange anchor $(date +%T) ==="; gen 8000 baseline 0.0001 1 ""               307 0 0 r2_c8_anchor.h5
echo "=== c9 pspl+flat    $(date +%T) ==="; gen 0 baseline 0.0001 1 ""                      308 34000 18000 r2_c9_pspl_flat.h5
echo "=== R2 SIM DONE $(date +%T) ==="
