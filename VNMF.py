import visualpackage as vis
import bestmapping as bp
import vatmap as vmp
import labelgen as lg
from PIL import Image
from sklearn.decomposition import NMF
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def VNMF(bow, n11, Tn, ntopics):
    model = NMF(ntopics, init='random', random_state=0)
    W = model.fit_transform(bow)
    H = model.components_
    DistE=squareform(pdist(W),'cosine')
    RI, ODM = vis.vat1(W,1,1)
    print(len(RI))
    visual = Image.fromarray(ODM)
    map = []
    map=vmp.vmap(n11,RI,Tn)
    gnd = lg.LBGEN(n11,ntopics)
    res1=bp.BestMap(gnd,map)
    return res1,gnd
