import visualpackage as vis
import bestmapping as bp
import vatmap as vmp
import labelgen as lg
from sklearn.metrics import confusion_matrix
from plsa import Corpus, Pipeline, Visualize
from plsa.pipeline import DEFAULT_PIPELINE
from plsa.algorithms import PLSA
import visualmvs as mvs
import MRMSTOPIC as mr

def VPLSI(n11, datafile,Tn, ntopics,samplesize):
    pipeline = Pipeline(*DEFAULT_PIPELINE)
    c1 = Corpus.from_csv(datafile, pipeline)
    print(c1)
    plsa = PLSA(c1, ntopics, True)
    result = plsa.fit()
    WPLSI=result.topic_given_doc
    RIPLSIE, ODMPLSIE, RIPLSIC, ODMPLSIC = vis.vat1(WPLSI,1,4)
    mapPLSI=vmp.vmap(n11,RIPLSIE,Tn,ntopics)
    gndPLSI = lg.LBGEN(n11, ntopics, Tn)
    res4=bp.BestMap(gndPLSI,mapPLSI)
    resultsPLSI = confusion_matrix(gndPLSI, res4)
    mapPLSIC = vmp.vmap(n11, RIPLSIC, Tn, ntopics)
    gndPLSIC = lg.LBGEN(n11, ntopics, Tn)
    res4C = bp.BestMap(gndPLSIC, mapPLSIC)
    resultsPLSI = confusion_matrix(gndPLSIC, res4C)
    #RIPLSImvs,ODMPLSImvs = mvs.visualmvs(Tn,WPLSI,4)
    RIPLSImrms, ODMmrms = mr.mrmstopic(Tn, WPLSI, 4, ntopics, samplesize)
    mapPLSImrms = vmp.vmap(n11, RIPLSImrms, Tn, ntopics)
    gndPLSImrms = lg.LBGEN(n11, ntopics, Tn)
    resPLSImrms = bp.BestMap(gndPLSImrms, mapPLSImrms)
    resultsPLSI = confusion_matrix(gndPLSIC, res4C)
    return res4, gndPLSI, res4C, gndPLSIC, resPLSImrms, gndPLSImrms,WPLSI