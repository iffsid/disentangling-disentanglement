#!/bin/bash

# This version simply makes a train-test split with identical parameters, but different
# number of data points.

# pinwheel generator from here:
# http://hips.seas.harvard.edu/content/synthetic-pinwheel-data-matlab

scriptdir=$(dirname "$0")
datadir="$scriptdir"/../data/

cat <<EOF | matlab -nosplash -nodesktop -nodisplay -nojvm
addpath('$scriptdir');
[tX, tY] = pinwheel(0.1, 0.3, 4, 100, 0.25);
[sX, sY] = pinwheel(0.1, 0.3, 4,  100, 0.25);
save('$datadir/pinwheel_small_train.mat', 'tX', 'tY');
save('$datadir/pinwheel_small_test.mat', 'sX', 'sY');
exit;
EOF