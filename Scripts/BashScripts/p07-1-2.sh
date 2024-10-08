# Run from within this directory

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../..

python -m Thesis-Lippmann-Schwinger.Scripts.p07_lse_solves 1 2 0
python -m Thesis-Lippmann-Schwinger.Scripts.p07_lse_solves 1 2 1
python -m Thesis-Lippmann-Schwinger.Scripts.p07_lse_solves 1 2 2
