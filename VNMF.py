import visualpackage as vis
import bestmapping as bp
import vatmap as vmp
import labelgen as lg
from PIL import Image
from sklearn.decomposition import NMF
import numpy as np
import visualmvs as mvs
import MRMSTOPIC as mr
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def VNMF(bow, n11, Tn, ntopics,samplesize):
    model = NMF(ntopics, init='random', random_state=0)
    W = model.fit_transform(bow)
    RIE, ODME,RIC,ODMC = vis.vat1(W,1,1)
    visual = Image.fromarray(ODME)
    map = []
    map=vmp.vmap(n11,RIE,Tn,ntopics)
    print(np.unique(map))
    gnd1 = lg.LBGEN(n11,ntopics,Tn)
    res1=bp.BestMap(gnd1,map)
    visual = Image.fromarray(ODME)
    map1 = []
    map1 = vmp.vmap(n11, RIC, Tn, ntopics)
    print(np.unique(map1))
    gnd2 = lg.LBGEN(n11, ntopics, Tn)
    res2 = bp.BestMap(gnd2, map)
    #RImvs,ODMmvs = mvs.visualmvs(Tn,W,1)
    RImrms,ODMmrms = mr.mrmstopic(Tn,W,1,ntopics,samplesize)
    map2 = []
    #map2 = vmp.vmap(n11, RImvs, Tn, ntopics)
    map3 =[]
    map3 = vmp.vmap(n11, RImrms, Tn, ntopics)
    #print(np.unique(map2))
    #gndNMFmvs = lg.LBGEN(n11, ntopics, Tn)
    #resNMFmvs = bp.BestMap(gndNMFmvs, map2)
    gndNMFmrms = lg.LBGEN(n11, ntopics, Tn)
    resNMFmrms = bp.BestMap(gndNMFmrms, map3)
    return res1,gnd1,res2,gnd2, resNMFmrms, gndNMFmrms,W
