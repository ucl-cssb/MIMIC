
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import math

num_species = 100

tmax = 100
sampling_time = 10
dt = 1
sampling_times = np.arange(0, tmax, sampling_time)
times = np.arange(0, tmax, dt)
batch_size = 32

errors = np.zeros((2,5,5,2))

# load data from first set of simulations
for exp in range(0, 100):
    tc, zp, sp = np.unravel_index(exp, ((4, 5, 5)))  # get indices into param arrays
    # inestigation scan over

    num_timecoursess = [96, 480, 1000, 5000]
    known_zero_props = [0, 0.25, 0.5, 0.75, 1.]
    dy_dx_regs = [1e3, 1e2, 1e1, 1., 1e-1]

    num_timecourses = num_timecoursess[tc]
    known_zero_prop = known_zero_props[zp]
    dy_dx_reg = dy_dx_regs[sp]

    try:
        path = './repeat' + str(exp+1)
        inputs = np.load(path + '/inputs.npy')
        preds = np.load(path + '/preds.npy')
        targets = np.load(path + '/targets.npy')
    except:
        print('error loading file')



    val_prop = 0.1
    split = int((1 - val_prop) * len(inputs))
    split = batch_size * math.floor(split / batch_size)
    train_inputs = inputs[:split]
    train_targets = targets[:split]


    train_preds = preds[:split, :, :]

    val_inputs = inputs[split:]
    val_targets = targets[split:]
    val_preds = preds[split:, :, :]

    print(train_preds.shape, val_preds.shape)

    train_loss = np.mean((train_targets - train_preds)**2)
    val_loss = np.mean((val_targets[:val_preds.shape[0]] - val_preds)**2)



    # get the loss of species that are actually rpesent in the sample as 0s throw off the loss

    d_ind_t, s_ind_t = np.nonzero(np.sum(train_targets, axis = 1)) # get the indices where a species is non zero in a timeseries
    d_ind_v, s_ind_v = np.nonzero(np.sum(val_targets[:val_preds.shape[0]], axis = 1)) # get the indices where a species is non zero in a timeseries

    train_loss_nz = np.mean((train_targets[d_ind_t, :, s_ind_t] - train_preds[d_ind_t, :, s_ind_t] ) ** 2)
    val_loss_nz = np.mean((val_targets[d_ind_v, :, s_ind_v]  - val_preds[d_ind_v, :, s_ind_v] ) ** 2)



    if num_timecourses in [96, 480]:
        print('tc', tc)
        errors[tc, zp, sp, 0] += train_loss_nz/3
        errors[tc, zp, sp, 1] += val_loss_nz/3


    # print(num_timecourses, known_zero_prop, species_prob, train_loss, val_loss)
    # for i in range(1):
    #
    #     plot_fit_gMLV_pert(np.vstack((inputs[-i,0,:num_species][np.newaxis,:],preds[-i,:,:])),  inputs[-i, :,num_species:], None, None, sampling_times,np.vstack((inputs[-i,0,:num_species][np.newaxis,:],targets[-i,:,:])) , times)
    #
    # # plt.plot(train_loss, label='train')
    # # plt.plot(val_loss, label='test')
    # # plt.legend()
    # # plt.xlabel('epoch')
    # # plt.ylabel('loss')
    # plt.show()
    # plt.close('all')

# load data from second set of simulations
for exp in range(0, 100):
    tc, zp, sp = np.unravel_index(exp, ((4, 5, 5)))  # get indices into param arrays
    # inestigation scan over

    num_timecoursess = [96, 480, 96, 480]
    known_zero_props = [0, 0.25, 0.5, 0.75, 1.]
    # species_probs = [0.1, 0.25, 0.5, 0.75, 1.]
    dy_dx_regs = [1e3, 1e2, 1e1, 1., 1e-1]

    num_timecourses = num_timecoursess[tc]
    known_zero_prop = known_zero_props[zp]
    dy_dx_reg = dy_dx_regs[sp]

    try:
        path = './repeat' + str(exp+1)
        inputs = np.load(path + '/inputs.npy')
        preds = np.load(path + '/preds.npy')
        targets = np.load(path + '/targets.npy')
    except:
        print('error loading file')



    val_prop = 0.1
    split = int((1 - val_prop) * len(inputs))
    split = batch_size * math.floor(split / batch_size)
    train_inputs = inputs[:split]
    train_targets = targets[:split]


    train_preds = preds[:split, :, :]

    val_inputs = inputs[split:]
    val_targets = targets[split:]
    val_preds = preds[split:, :, :]

    print(train_preds.shape, val_preds.shape)

    train_loss = np.mean((train_targets - train_preds)**2)
    val_loss = np.mean((val_targets[:val_preds.shape[0]] - val_preds)**2)



    # get the loss of species that are actually rpesent in the sample as 0s throw off the loss

    d_ind_t, s_ind_t = np.nonzero(np.sum(train_targets, axis = 1)) # get the indices where a species is non zero in a timeseries
    d_ind_v, s_ind_v = np.nonzero(np.sum(val_targets[:val_preds.shape[0]], axis = 1)) # get the indices where a species is non zero in a timeseries

    train_loss_nz = np.mean((train_targets[d_ind_t, :, s_ind_t] - train_preds[d_ind_t, :, s_ind_t] ) ** 2)
    val_loss_nz = np.mean((val_targets[d_ind_v, :, s_ind_v]  - val_preds[d_ind_v, :, s_ind_v] ) ** 2)

    tc = 0 if num_timecourses == 96 else 1

    print('tc', tc)
    errors[tc, zp, sp, 0] += train_loss_nz/3
    errors[tc, zp, sp, 1] += val_loss_nz/3

for tc in range(0, 2):

    plt.subplot(2,2,tc+1)
    plt.title('num timecourses ' + str(num_timecoursess[tc]))
    plt.imshow(errors[tc, :, :, 1])
    plt.ylabel('known zeros prob')
    plt.yticks([0,1,2,3], known_zero_props)
    plt.xlabel('dy_dx_reg')
    plt.xticks([0, 1, 2, 3], dy_dx_regs)
    plt.colorbar()

    print(num_timecoursess[tc], errors[tc, :, :, 1])

    print(inputs.shape)




plt.show()