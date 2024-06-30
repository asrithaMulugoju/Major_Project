import visualpackage as vis
import bestmapping as bp
import vatmap as vmp
import labelgen as lg
import numpy as np
from gensim.models.lsimodel import LsiModel
from gensim.models.tfidfmodel import TfidfModel
import visualmvs as mvs
import MRMSTOPIC as mr

def VLSI(n11,corpus,dictionary, Tn, ntopics,samplesize):
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = LsiModel(corpus_tfidf, id2word = dictionary, num_topics = ntopics)
    corpus_lsi = lsi[corpus_tfidf]
    #for doc in corpus_lsi:
    # print(doc)

    r1, c1 = (Tn, ntopics)
    WLSI = np.zeros((Tn, ntopics))
    for i in range(0,Tn):
        for j in range(0,ntopics):
            b = corpus_lsi[i][j][1]
            WLSI[i][j] = b
    RILSIE, ODMLSIE, RILSIC, ODMLSIC = vis.vat1(WLSI,1,3)
    mapLSI=vmp.vmap(n11,RILSIE,Tn,ntopics)
    gndLSI = lg.LBGEN(n11, ntopics, Tn)
    res3=bp.BestMap(gndLSI,mapLSI)
    mapLSIC = vmp.vmap(n11, RILSIC, Tn, ntopics)
    gndLSIC = lg.LBGEN(n11, ntopics, Tn)
    res3C = bp.BestMap(gndLSIC, mapLSIC)
    RImrms, ODMmrms = mr.mrmstopic(Tn, WLSI, 3, ntopics, samplesize)
    #RImvs, ODMmvs = mvs.visualmvs(Tn, WLSI,3)
    mapLSImrms = vmp.vmap(n11, RImrms, Tn, ntopics)
    gndLSImrms = lg.LBGEN(n11, ntopics, Tn)
    resLSImrms = bp.BestMap(gndLSImrms, mapLSImrms)
    return res3, gndLSI, res3C, gndLSIC, resLSImrms, gndLSImrms, WLSI