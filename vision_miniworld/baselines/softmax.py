import numpy as np

# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

scores = [3.0, 1.0, -0.5]
probs = softmax(scores)
inte = np.array([.5,.5,39.])

# import pdb;pdb.set_trace()

# print(probs)
# print(np.log(probs))
# print(softmax(np.log(probs)))
# print(softmax(np.log(softmax(np.log(probs)))) )
pi_I = probs * inte / sum(probs*inte)
print(pi_I)
print(np.log(pi_I))
print(softmax(np.log(pi_I)) )

# probs_I = softmax(scores * np.log(inte/(1-inte)) )
# print(probs_I)