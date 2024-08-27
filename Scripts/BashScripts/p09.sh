# Run from within this directory

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../..

python -m Lippmann-Schwinger.Scripts.p09_plot_images 0 3
python -m Lippmann-Schwinger.Scripts.p09_plot_images 1 3
python -m Lippmann-Schwinger.Scripts.p09_plot_images 2 3
