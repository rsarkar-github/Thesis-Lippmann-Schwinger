# Run from within this directory

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../..

python -m Lippmann-Schwinger.Scripts.p11_initial_vel_solves 2 2 0
python -m Lippmann-Schwinger.Scripts.p11_initial_vel_solves 2 2 1
python -m Lippmann-Schwinger.Scripts.p11_initial_vel_solves 2 2 2
