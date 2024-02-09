import visualpackage as vis
import bestmapping as bp
import vatmap as vmp
import labelgen as lg
from sklearn.metrics import confusion_matrix
from plsa import Corpus, Pipeline, Visualize
from plsa.pipeline import DEFAULT_PIPELINE
from plsa.algorithms import PLSA

def VPLSI(n11, datafile,Tn, ntopics):
    pipeline = Pipeline(*DEFAULT_PIPELINE)
    c1 = Corpus.from_csv(datafile, pipeline)
    plsa = PLSA(c1, ntopics, True)
    result = plsa.fit()
    WPLSI=result.topic_given_doc
    RIPLSI, ODMPLSI = vis.vat1(WPLSI,1,4)
    mapPLSI=vmp.vmap(n11,RIPLSI,Tn)
    gndPLSI = lg.LBGEN(n11, ntopics)
    res4=bp.BestMap(gndPLSI,mapPLSI)
    resultsPLSI = confusion_matrix(gndPLSI, res4)
    return res4, gndPLSI