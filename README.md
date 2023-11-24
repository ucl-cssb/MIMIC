Usage (on command line):

`python3 run_gMLV_sims.py <output directory> <number to simulate>`

for example:
`python run_gMLV_sims.py "C:\\Users\\User\\Desktop\\test_GMLV" 100`


The method for generating the gLV simulations is from this paper (it is in the code gMLV/gMLV_sim.py):

https://onlinelibrary.wiley.com/doi/full/10.1002/bies.201600188



########## Installing conda environment ##########
```
conda create -n gMLV python=3.10
conda activate gMLV
conda install tensorflow, etc.

```

Notes: no need to install 'casadi' or ODE.RED