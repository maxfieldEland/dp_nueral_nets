# dp_nueral_nets
Repository for Data Privacy Final Project. Purpose of project is to investigate the 
possibility to retaining neural network performance when trained on synthetic data 
derived by unary encoding. Contributors include : Sophia, Nick and Max. 


We use the open source MNIST Data set which can be found at : http://yann.lecun.com/exdb/mnist/


# models
## unary encoding with clustering
- convert each pixel to binary: either on or off
- convert each label to one hot representation
- use unary encoding on each example to get local synthetic dataset
- We hope that the noisy images are not too bad
- but the noisy labels are very bad and not recoverable on their own
- use unsupervised clustering on the examples (ignoring labels)
- hopefully these form good clusters
- then look at all of the noisy labels within each cluster, and determine the most likely label for that cluster (we are thinking that this will be very evident because each cluster should have a LOT of examples in it, so averaging is on our side)
- Then we label the clusters, and we have a differentially private model to take an image and classify it
- look at performance

## marginals and convolution neural net
- build 2-way marginals between each pixel and the label
- build 1-way marginal for the label
- build synthetic dataset by drawing a label from the 1-way marginal, then draw a pixel value from each 2-way marginal, given that label
- train CNN classify the data
- look at performance
