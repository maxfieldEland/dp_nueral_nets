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
    return [perturb_example(ex, p, q) for ex in binary_train]

def perturb_label(label, p, q):
    domain  = np.arange(0, 10)
    encoded = [1 if label == d else 0 for d in domain]
    return np.array([perturb_bit(b, p, q) for b in encoded])

def perturb_labels(labels, p, q):
    return np.array([perturb_label(l, p, q) for l in labels])


def gaussian_mech_vec(v, sensitivity, epsilon, delta):
    return v + np.random.normal(loc=0, scale=sensitivity * np.sqrt(2*np.log(1.25/delta)) / epsilon, size=len(v))

def gaussian_example(example, epsilon=1, delta=1/((28*28)**2)): # 1/n^2
    dims = example.shape

    # normalize x so it has a norm of 1
    x = example.reshape(-1)
    x = x / np.linalg.norm(x)
    # print(np.linalg.norm(x, ord=2))
    x = gaussian_mech_vec(x, 1, epsilon, delta)
    x = np.clip(x, 0, None)
    return x.reshape(dims)

def gaussian_examples(examples, epsilon=1, delta=1/((28*28)**2)):
    return [gaussian_example(example, epsilon, delta) for example in examples]
    
    
def rr(pixel, cutoff, p, q):
    # if np.random.rand() < 0.5:
        # return pixel > cutoff
    # else:
        # if np.random.rand() < 0.5:
            # return True
        # else:
            # return False
    if pixel > cutoff:
        return np.random.rand() < p
    else:
        return np.random.rand() < q
            

def ue_eps(p, q):
    return np.log((p*(1-q)) / (q*(1-p)))

def rr_ex(example, p, q):
    dims = example.shape
    xs = example.reshape(-1)
    xs_rr = np.array([rr(x, 115, p, q) for x in xs])
    epsilon = ue_eps(p, q)
    print(f"Epsilon={epsilon}")
    return xs_rr.reshape(dims)
    

    
train_images=np.load('data/train_images.npy')
train_labels=np.load('data/train_labels.npy')
test_images=np.load('data/test_images.npy')
test_labels = np.load('data/test_labels.npy')

#x = train_images[0,:,:]
#
#x_local = gaussian_example(x,1,1/((28*28)**2) )
#x_local = gaussian_example(x,150,1/((28*28)**2) )
#eps = [10,150,500,3000,5000,200000]
#
#
#
#fig = plt.figure(figsize = (8,12))
#for idx,ep in enumerate(eps):
#    x_local = gaussian_example(x,ep,1/((28*28)**2))
#    ax = fig.add_subplot(3,2,idx+1,label = "HIII")
#    print(ep)
#    ax.set_title('epsilon = '+str(ep))
#    plt.imshow(x_local)
#    plt.axis('off')
#
#plt.savefig("epsilons.png", bbox_inches='tight')
#plt.show()
#
#x_local = gaussian_example(x,150,1/((28*28)**2) )
#
#
#show_examples([gaussian_example(ex, epsilon=150) for ex in train_images[:12]])
#




#np.save('data/train_images.npy', train_images)
#np.save('data/test_images.npy', test_images)
#np.save('data/train_labels.npy', train_labels)
#np.save('data/test_labels.npy', test_labels)




# todo: make figures of possible cutoffs for writeup. or maybe not
# because looking at these is not DP?

show_cutoff_plots = False
if show_cutoff_plots:
    possible_cutoffs = np.linspace(0,254, 10)
    visualize_cutoff(train_images[:12], possible_cutoffs)

# around 115 looks good, so let's use that
binary_train = train_images > 0

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

# save perturbed data to data folder
np.save('data/local_train_images.npy', local_train_images)
np.save('data/local_train_labels.npy', local_train_labels)






