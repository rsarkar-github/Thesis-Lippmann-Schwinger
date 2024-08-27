# Run from within this directory

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../..

python -m Lippmann-Schwinger.Scripts.p05_calculate_green_func 0 0
python -m Lippmann-Schwinger.Scripts.p05_calculate_green_func 0 1
python -m Lippmann-Schwinger.Scripts.p05_calculate_green_func 0 2
python -m Lippmann-Schwinger.Scripts.p05_calculate_green_func 0 3
python -m Lippmann-Schwinger.Scripts.p05_calculate_green_func 1 0
python -m Lippmann-Schwinger.Scripts.p05_calculate_green_func 1 1
python -m Lippmann-Schwinger.Scripts.p05_calculate_green_func 1 2
python -m Lippmann-Schwinger.Scripts.p05_calculate_green_func 1 3
python -m Lippmann-Schwinger.Scripts.p05_calculate_green_func 2 0
python -m Lippmann-Schwinger.Scripts.p05_calculate_green_func 2 1
python -m Lippmann-Schwinger.Scripts.p05_calculate_green_func 2 2
python -m Lippmann-Schwinger.Scripts.p05_calculate_green_func 2 3