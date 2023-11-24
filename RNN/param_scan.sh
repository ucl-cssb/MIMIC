#!/bin/sh

for L2_reg in 1e-8 1e-7 1e-6 1e-5
do
  #for GRU_size in 32 64 128 256 512
  for GRU_size in 256 512
  do
    for dy_dx_reg in 1e-2 1e-3 1e-4
    do
      python RNN.py $L2_reg $GRU_size $dy_dx_reg
    done
  done
done

