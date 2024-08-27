# Run from within this directory

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../..

python -m Lippmann-Schwinger.Scripts.p01a_helmholtz_radial_performance 5.0
python -m Lippmann-Schwinger.Scripts.p01a_helmholtz_radial_performance 7.5
python -m Lippmann-Schwinger.Scripts.p01a_helmholtz_radial_performance 10.0
