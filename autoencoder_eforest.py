#!/usr/bin/env python
# coding: utf-8

# In[ ]:


""""
This code is based on theoretically https://arxiv.org/abs/1709.09018 ,and technically https://github.com/AntoinePassemiers/Encoder-Forest
Please download "encoder" from https://github.com/AntoinePassemiers/Encoder-Forest by Antoine Passemiers
""""

from encoder import EncoderForest
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

encoder = EncoderForest(1000)
encoder.fit(x_train, max_depth=20) # Fit embedding trees (unsupervised eForest)
test_encoded=encoder.encode(x_test)
encoded = encoder.encode(x_train) # Encode the entire training set


save_decoded=[]
for j in range (10):
    print(j)
    decoded = encoder.decode(test_encoded[j]) # Decode 

    # Intuitively, the performance of an embedding tree could be measured
    # as the log(volume) returned by this tree divided by the log(volume) of the MCR.

    # Get the path rules used to decode the first instance
    rule_list = encoder.compute_rule_list(test_encoded[j])

    # For each path rule (hyper-parallelepiped), compute its log(volume)
    for i, path_rule in enumerate(rule_list):
        log_volume = path_rule.compute_volume()
        #print("Log-volume of hyper-parallelepiped %i (tree %i): %f" % (i, i, log_volume))

    # Get the intersection of the subspaces described by the path rules
    MCR = encoder.calculate_MCR(rule_list)

    # Compute the log(volume) of the subspace described by the MCR
    #print("MCR log-volume: %f" % MCR.compute_volume())

    # Decode by generating a random sample in the final subspace
    decoded = MCR.sample()
    save_decoded.append(decoded)
    
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

for i in range(len(save_decoded)):
    plt.figure(figsize=(10, 4))  # Figure size

    # Plot for original image
    plt.subplot(1, 2, 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Plot for decoded image
    plt.subplot(1, 2, 2)
    plt.imshow(save_decoded[i].reshape(28, 28), cmap='gray')
    plt.title('Decoded Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# In[ ]:




