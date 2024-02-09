import visualpackage as vis
import bestmapping as bp
import vatmap as vmp
import labelgen as lg
import numpy as np
from gensim.models.lsimodel import LsiModel
from gensim.models.tfidfmodel import TfidfModel

def VLSI(n11,corpus,dictionary, Tn, ntopics):
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = LsiModel(corpus_tfidf, id2word = dictionary, num_topics = ntopics)
    corpus_lsi = lsi[corpus_tfidf]
    #for doc in corpus_lsi:
    # print(doc)
    r1, c1 = (Tn, ntopics)
    WLSI = np.zeros((Tn, ntopics))
    for i in range(0,Tn):
        a = corpus_lsi[i][0][1]
        WLSI[i][0] = a
        b = corpus_lsi[i][1][1]
        WLSI[i][1] = b
#print(WLSI)
    RILSI, ODMLSI = vis.vat1(WLSI,1,3)
    mapLSI=vmp.vmap(n11,RILSI,Tn)
    gndLSI = lg.LBGEN(n11, ntopics)
    res3=bp.BestMap(gndLSI,mapLSI)
    return res3, gndLSI