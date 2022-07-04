import numpy as np
import matplotlib.pyplot as plt

print(np.unravel_index(199, ((2,4,5,5))))
sys.exit()

L2_regs = [1e-8, 1e-7, 1e-6, 1e-5]
GRU_sizes = [32, 64, 128, 256]
dy_dx_regs = [1e-2, 1e-3, 1e-4]

for dy_dx_reg in dy_dx_regs:
    plt.figure()
    plt.title(str(dy_dx_reg))
    i = 1
    for L2_reg in L2_regs:

        for GRU_size in GRU_sizes:
            plt.subplot(4, 4, i)
            train_loss = np.load('/home/neythen/Desktop/Projects/gMLV/RNN/param_scan_040722/train_loss'+str(L2_reg) + str(GRU_size) + str(dy_dx_reg) + '.npy')
            val_loss = np.load('/home/neythen/Desktop/Projects/gMLV/RNN/param_scan_040722/val_loss'+str(L2_reg) + str(GRU_size) + str(dy_dx_reg) + '.npy')

            print(L2_reg, GRU_size, dy_dx_reg, train_loss[-1],  val_loss[-1])
            plt.title(str(L2_reg) + str(GRU_size) + str(dy_dx_reg))
            plt.plot(train_loss)
            plt.plot(val_loss)
            i+=1

plt.close('all')

for dy_dx_reg in dy_dx_regs:



    losses = np.zeros((4, 4, 2))

    for i, L2_reg in enumerate(L2_regs):

        for j, GRU_size in enumerate(GRU_sizes):

            train_loss = np.load('/home/neythen/Desktop/Projects/gMLV/RNN/param_scan_040722/train_loss'+str(L2_reg) + str(GRU_size) + str(dy_dx_reg) + '.npy')
            val_loss = np.load('/home/neythen/Desktop/Projects/gMLV/RNN/param_scan_040722/val_loss'+str(L2_reg) + str(GRU_size) + str(dy_dx_reg) + '.npy')

            losses[i,j, 0] = train_loss[-1]
            losses[i,j, 1] = val_loss[-1]

    # plt.figure()
    # plt.title(str(dy_dx_reg) + 'train')
    # plt.imshow(losses[:, :, 0])
    # plt.colorbar()


    fig, ax = plt.subplots(1,1)
    ax.set_title('Validation loss ' + str(dy_dx_reg) + 'dy_dx reg')
    img = ax.imshow(losses[:, :, 1], vmin = 0, vmax = 0.00012)
    ax.set_xticks([0,1,2,3])
    ax.set_yticks([0,1,2,3])
    ax.set_yticklabels(L2_regs)
    ax.set_xticklabels(GRU_sizes)

    ax.set_xlabel('GRU size')
    ax.set_ylabel('L2 reg')
    fig.colorbar(img)


plt.show()