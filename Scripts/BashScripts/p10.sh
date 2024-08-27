# Run from within this directory

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../..

python -m Lippmann-Schwinger.Scripts.p10_plot_images_sigsbee_15Hz_Helmholtz
