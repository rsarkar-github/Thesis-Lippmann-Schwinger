# Lippmann-Schwinger

- Numpy, Devito based Python codes for the Lippmann-Schwinger problem

- Contains all codes to reproduce figures and tables in Part I: Chapter 2 of thesis

# How to reproduce the results? The following steps were tested on Linux (Centos & Ubuntu) only.
- After cloning this repository, install the Python packages as indicated in the file "requirements.txt".
Steps to install the packages are indicated below in the Section **How to install the packages**.

- Navigate to the directory Lippmann-Schwinger/Scripts/BashScripts/

- Run all the scripts serially

- For example, to run **p01.sh**, just type the command
```ruby
bash p01.sh
```

- The above runs will create some files in the directory **Lippmann-Schwinger/Data/**, and all figures will be 
created in the directory **Lippmann-Schwinger/Fig/**.

# How to install the packages (assumes conda is installed).

```ruby
conda create -n py39 python=3.9
conda activate py39
conda install -c anaconda numpy=1.23.5
conda install -c anaconda scipy=1.10.0
conda install -c numba numba=0.57.1
conda install matplotlib=3.7.1
conda install tqdm

```