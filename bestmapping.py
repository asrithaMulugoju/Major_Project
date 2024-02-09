import numpy as np
import hg as h1
def BestMap(L1, L2):

    #L1 = L1.flatten(order='F').astype(float)
    #L2 = L2.flatten(order='F').astype(float)
    if L1.__len__() != L2.__len__():
        print('size(L1) must == size(L2)')
    Label1 = np.unique(L1)
    nClass1 = Label1.__len__()
    Label2 = np.unique(L2)
    nClass2 = Label2.__len__()
    nClass = max(nClass1, nClass2)
    # For Hungarian - Label2 are Workers, Label1 are Tasks.
    G = np.zeros([nClass, nClass])
    for i in range(0, nClass2):
        for j in range(0, nClass1):
            G[i, j] = np.sum(np.logical_and(L2 == Label2[i], L1 == Label1[j]))

    c = h1.Hungarian(-G)
    print(c)
    newL2 = np.zeros(L2.__len__())
    for i in range(0, nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2.astype(int)
