
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt



errors = np.zeros((4,5,5,2))
for exp in range(0, 100):
    tc, zp, sp = np.unravel_index(exp, ((4, 5, 5)))  # get indices into param arrays
    # inestigation scan over

    num_timecoursess = [100, 500, 1000, 5000]
    known_zero_props = [0, 0.25, 0.5, 0.75, 1.]
    species_probs = [0.1, 0.25, 0.5, 0.75, 1.]

    num_timecourses = num_timecoursess[tc]
    known_zero_prop = known_zero_props[zp]
    species_prob = species_probs[sp]

    path = '/Users/neythen/Desktop/Projects/gMLV/RNN/cluster_results/repeat' + str(exp+1)

    inputs = np.load(path + '/inputs.npy')
    preds = np.load(path + '/preds.npy')
    targets = np.load(path + '/targets.npy')

    val_prop = 0.1
    split = int((1 - val_prop) * len(inputs))
    train_inputs = inputs[:split]
    train_targets = targets[:split]
    train_preds = preds[:split]

    val_inputs = inputs[split:]
    val_targets = targets[split:]
    val_preds = preds[split:]


    train_loss = np.mean((train_targets - train_preds)**2)
    val_loss = np.mean((val_targets - val_preds)**2)


    errors[tc, zp, sp, 0] = train_loss
    errors[tc, zp, sp, 1] = val_loss


    print(num_timecourses, known_zero_prop, species_prob, train_loss, val_loss)

for tc in range(0, 4):
    plt.figure()
    plt.title('num timecourses ' + str(num_timecoursess[tc]))
    plt.imshow(errors[tc, :, :, 1], vmax = 0.015)
    plt.ylabel('known zeros prob')
    plt.yticks([0,1,2,3], known_zero_props)
    plt.xlabel('species prob')
    plt.xticks([0, 1, 2, 3], species_probs)
    plt.colorbar()

    print(num_timecoursess[tc], errors[tc, :, :, 1])

plt.show()