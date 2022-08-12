import numpy as np
from casadi import *
import numpy as np
import matplotlib.pyplot as plt

actual_params = np.load('../../working_dir/generated_params.npy')
y0 = np.load('../../working_dir/generated_y0.npy')

titles = ['Random 10 days', 'OED 10 days', 'Random 100 days', 'OED 100 days']
for i,dir in enumerate(['rand_ten_days', 'MPC_OED_ten_days']):
    actual_params = np.load('../../working_dir/generated_params.npy')
    inferred = np.load(dir + '/all_final_params_opt.npy')
    losses = np.load(dir + '/all_losses_opt.npy')
    print('loss shape', losses.shape)
    param_error = np.sum(np.abs(((inferred- actual_params)/actual_params)))/(len(inferred)*len(inferred[0]))
    print('total error: ', param_error)
    print('total loss', np.mean(losses))
    inferred/=actual_params
    actual_params/= actual_params

    print(inferred.shape, actual_params.shape)

    cov = np.cov(inferred.T)
    mean_inferred = np.mean(inferred, axis = 0)

    #[print(actual_params[i], mean_inferred[i], 1.96*sqrt(cov[i][i])/sqrt(30)) for i in range(len(actual_params))]

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

    #plt.subplot(1,2,i+1)

    plt.errorbar(range(1,len(actual_params)+1), mean_inferred, yerr = ci, fmt = 'x', label = titles[i] + ' 95% CI')
    plt.scatter(range(1,len(actual_params)+1),actual_params, color = 'green', label = 'True Value')
    plt.legend()
    plt.title(titles[i])
    plt.ylim([.50, 1.5])
    plt.xticks(range(1,len(actual_params)+1))
    plt.xlabel('Parameter number')
    plt.ylabel('Parameter value')
    plt.text(0.5, 0.5, 'Average parameter error: {0:.2f} \nlog(det(COV)): {1:.2f} \n'.format(param_error, logdet_cov))


plt.show()