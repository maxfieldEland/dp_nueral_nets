import numpy as np
from mnist_utils import load_mnist
import matplotlib.pyplot as plt

def show_examples(examples, title="", n_col=4):
    fig, axs = plt.subplots(int(len(examples) / n_col), n_col)
    for ax, example in zip(axs.reshape(-1), examples):
        ax.imshow(example)
    fig.suptitle(title)
    fig.show()
    
    

def visualize_cutoff(examples, cutoffs):
    for c in cutoffs:
        show_examples(examples > c, f"cutoff = {c:.2f}")


def perturb_bit(bit, p, q):
    sample = np.random.random()

    if bit == 1:
        if sample <= p:
            return 1
        else:
            return 0
    elif bit == 0:
        if sample <= q:
            return 1
        else: 
            return 0

def perturb_example(example, p, q):
    dims = example.shape
    perturbed_list = [perturb_bit(b, p, q) for b in example.reshape(-1)]
    return np.array(perturbed_list).reshape(dims)

def privacy_cost(p, q):
    return np.log((p * (1-q)) / (q * (1-p)))

def visualize_eps(examples, p, q):
    epsilon = privacy_cost(p, q)
    show_examples(examples, f"$\epsilon = {epsilon:.3f}$, $p={p:.3f}$, $q={q:.3f}$")


def perturb_examples(examples, p, q):
    return [perturb_example(ex, p, q) for ex in binary_train[:12]]

def perturb_label(label, p, q):
    domain  = np.arange(0, 10)
    encoded = [1 if label == d else 0 for d in domain]
    return np.array([perturb_bit(b, p, q) for b in encoded])

def perturb_labels(labels, p, q):
    return np.array([perturb_label(l, p, q) for l in labels])


    
train_images, train_labels, test_images, test_labels = load_mnist()

np.save('data/train_images.npy', train_images)
np.save('data/test_images.npy', test_images)
np.save('data/train_labels.npy', train_labels)
np.save('data/test_labels.npy', test_labels)




# todo: make figures of possible cutoffs for writeup. or maybe not
# because looking at these is not DP?

show_cutoff_plots = False
if show_cutoff_plots:
    possible_cutoffs = np.linspace(0,254, 10)
    visualize_cutoff(train_images[:12], possible_cutoffs)

# around 115 looks good, so let's use that
binary_train = train_images > 115

# look at different values of p and q. Assume q = 1-p. Could change this? look into literature 
show_p_q_plots = False
if show_p_q_plots:
    ps_and_qs = [(p, 1-p) for p in np.linspace(0.501, 0.999, 10)]
    for p, q in ps_and_qs:
        visualize_eps(perturb_examples(binary_train[:12], p, q), p, q)


p = 0.88
q = 0.12

local_train_images = perturb_examples(binary_train, p, q)
local_train_labels = perturb_labels(train_labels, p, q)

# save perturbed data to data foulder
np.save('data/local_train_images.npy', local_train_images)
np.save('data/local_train_labels.npy', local_train_labels)






