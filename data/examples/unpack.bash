#!/bin/bash

#
# Unpacks data for examples
#

# navigate to mtuq/data
cd $(dirname ${BASH_SOURCE[0]})
wd=$PWD

for filename in \
    20090407201255351.tgz 20210809074550.tgz 20SPECFEM3D_SGT.tgz SPECFEM3D_SAC.tgz;
do
    cd $wd
    cd $(dirname $filename)
    echo "Unpacking $filename"
    tar -xzf $filename
done
echo "Done"
echo ""

