import numpy as np
from casadi import *
import numpy as np
import matplotlib.pyplot as plt

actual_params = np.load('../../working_dir/generated_params.npy')
y0 = np.load('../../working_dir/generated_y0.npy')

titles = ['Random 10 days', 'OED 10 days', 'Random 100 days', 'OED 100 days']
for i,dir in enumerate(['rand_ten_days', 'MPC_OED_ten_days',  'rand_100_days', 'MPC_OED_100_days']):

    inferred = np.load(dir + '/all_final_params_opt.npy')
    losses = np.load(dir + '/all_losses_opt.npy')
    print('loss shape', losses.shape)
    param_error = np.sum(((inferred- actual_params)/actual_params)**2)
    print('total error: ', param_error)
    print('total loss', np.mean(losses))
    #inferred/=actual_params

    print(inferred.shape, actual_params.shape)

    cov = np.cov(inferred.T)
    mean_inferred = np.mean(inferred, axis = 0)

    [print(actual_params[i], mean_inferred[i], 1.96*sqrt(cov[i][i])/sqrt(30)) for i in range(len(actual_params))]

    ci = [1.96*sqrt(cov[i][i])/sqrt(30) for i in range(len(actual_params))]

    q, r = qr(cov)

    det_cov = np.prod(diag(r).elements())

    logdet_cov = trace(log(r)).elements()[0]

    #print(check_symmetric(cov))
    #print(check_symmetric(cov))
    print('cov shape: ', cov.shape)

    print(' det cov: ', det_cov)
    print('log det cov; ',logdet_cov)
    print()

    plt.subplot(2,2,i+1)
    plt.scatter(range(1,len(actual_params)+1),actual_params, label = 'True Value')
    plt.errorbar(range(1,len(actual_params)+1), mean_inferred, yerr = ci, fmt = 'x', color = 'orange', label = '95% CI')
    plt.legend()
    plt.title(titles[i])
    plt.ylim([-1.25, 1.25])
    plt.xticks(range(1,len(actual_params)+1))
    plt.xlabel('Parameter number')
    plt.ylabel('Parameter value')
    #plt.text(0.5, 0.5, 'Total parameter error: {0:.2f} \nTotal prediction error: {1:.2f} \nlog(det(COV)): {2:.2f} \n'.format(param_error, np.mean(losses), logdet_cov))


plt.show()