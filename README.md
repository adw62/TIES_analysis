# TIES_analysis
Make analysis of TIES output from NAMD2/3 or OpenMM

# Installation and running

conda create --name ti_ana

conda activate ti_ana

unzip TIES_analysis-main.zip

cd TIES_analysis-main

sudo apt install python3-pip

pip install numpy==1.20.0

pip install pymbar==3.0.3

pip install scikit-learn==0.24.2

cd example

python3 ../ties_analysis/ties_analysis.py
