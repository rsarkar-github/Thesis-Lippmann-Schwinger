# Run from within this directory

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../..

python -m Thesis-Lippmann-Schwinger.Scripts.p08a_helmholtz_solves 1 3 1
python -m Thesis-Lippmann-Schwinger.Scripts.p08a_helmholtz_solves 1 3 2
