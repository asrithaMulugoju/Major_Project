import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
def vat1(X, return_ODM=False, I1 = False):
    """VAT means Visual assesement of tendency. basically, it allow to asses cluster tendency
    through a map based on the dissimiliraty matrix.

    Parameters
    ----------

    X : matrix
        numpy array

    return_ODM : return the Ordered Dissimalirity Matrix
        boolean (default to False)

    figuresize : size of the VAT.
        tuple (default to (10,10))


    Return
    -------

    ODM : matrix
        the ordered dissimalarity matrix plotted.

    """

    P, ODM = ordered_dissimilarity_matrix(X)
    fig = plt.figure()
    if I1==1:
        x = "VAT-NMF-E" + ".jpg"
    if I1==2:
        x="VAT-LDA-E" + ".jpg"
    if I1 == 3:
        x = " VAT-LSI-E" + ".jpg"
    if I1 == 4:
        x = " VAT-PLSI-E" + ".jpg"
#    _, ax = plt.subplots(figsize=figuresize)
    plt.imshow(ODM, cmap='gray', vmin=0, vmax=np.max(ODM))
    fig.savefig(x)
    if return_ODM == True:

        return P, ODM

def ordered_dissimilarity_matrix(X):
    """The ordered dissimilarity matrix is used by visual assesement of tendency. It is a just a a reordering
    of the dissimilarity matrix.


    Parameters
    ----------

    X : matrix
        numpy array

    Return
    -------

    ODM : matrix
        the ordered dissimalarity matrix .

    """

    # Step 1 :

    I = []

    R = pairwise_distances(X)
    P = np.zeros(R.shape[0], dtype="int")

    argmax = np.argmax(R)

    j = argmax % R.shape[1]
    i = argmax // R.shape[1]

    P[0] = i
    I.append(i)

    K = np.linspace(0, R.shape[0] - 1, R.shape[0], dtype="int")
    J = np.delete(K, i)

    # Step 2 :

    for r in range(1, R.shape[0]):

        p, q = (-1, -1)

        mini = np.max(R)

        for candidate_p in I:
            for candidate_j in J:
                if R[candidate_p, candidate_j] < mini:
                    p = candidate_p
                    q = candidate_j
                    mini = R[p, q]

        P[r] = q
        I.append(q)

        ind_q = np.where(np.array(J) == q)[0][0]
        J = np.delete(J, ind_q)

    # Step 3

    ODM = np.zeros(R.shape)

    for i in range(ODM.shape[0]):
        for j in range(ODM.shape[1]):
            ODM[i, j] = R[P[i], P[j]]

    # Step 4 :

    return P,ODM
