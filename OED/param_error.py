import numpy as np
from casadi import *

actual_params = np.load('working_dir/generated_params.npy')
y0 = np.load('working_dir/generated_y0.npy')


for dir in ['rand_ten_days', 'rand_100_days', 'MPC_OED_ten_days', 'MPC_OED_100_days']:

    inferred = np.load(dir + '/all_final_params_opt.npy')
    print((((inferred- actual_params)/actual_params)**2).shape)
    print('total error: ', np.sum(((inferred- actual_params)/actual_params)**2))
    inferred/=actual_params



    cov = np.cov(inferred.T)

    q, r = qr(cov)

    det_cov = np.prod(diag(r).elements())

    logdet_cov = trace(log(r)).elements()[0]
    print(cov)
    #print(check_symmetric(cov))
    #print(check_symmetric(cov))
    print('cov shape: ', cov.shape)

    print(' det cov: ', det_cov)
    print('log det cov; ',logdet_cov)