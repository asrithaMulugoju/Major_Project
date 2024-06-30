import visualpackage as vis
import bestmapping as bp
import vatmap as vmp
import labelgen as lg
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models
from gensim.sklearn_api import LdaTransformer
import visualmvs as mvs
import MRMSTOPIC as mr

def VLDA(n11,tweets,Tn, ntopics,samplesize):
    tokenizer = RegexpTokenizer(r'\w+')
    texts = []
    for i in range(0,Tn):
        texts.append(tokenizer.tokenize(tweets['t1'][i]))
    #VLDA Model
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    model = LdaTransformer(num_topics=ntopics, id2word=dictionary, iterations=20, random_state=1)
    WLDA= model.fit_transform(corpus)
    RILDA1, ODMLDA1, RILDAC,ODMLAC = vis.vat1(WLDA,1,2)
    mapLDA=vmp.vmap(n11,RILDA1,Tn,ntopics)
    gndLDA = lg.LBGEN(n11, ntopics, Tn)
    res2=bp.BestMap(gndLDA,mapLDA)
    mapLDA1 = vmp.vmap(n11, RILDAC, Tn, ntopics)
    gndLDAC = lg.LBGEN(n11, ntopics, Tn)
    resC = bp.BestMap(gndLDAC, mapLDA1)
    RImrms, ODMmrms = mr.mrmstopic(Tn, WLDA, 2, ntopics, samplesize)
    mapLDAmrms = vmp.vmap(n11, RImrms, Tn, ntopics)
    gndLDAmrms = lg.LBGEN(n11, ntopics, Tn)
    resLDAmrms = bp.BestMap(gndLDAmrms, mapLDAmrms)
    #RImvs, ODMmvs = mvs.visualmvs(Tn, WLDA,2)
    #mapLDAmvs = vmp.vmap(n11, RImvs, Tn, ntopics)
    #gndLDAmvs = lg.LBGEN(n11, ntopics, Tn)
    #resLDAmvs = bp.BestMap(gndLDAmvs, mapLDAmvs)
    return res2, gndLDA, corpus, dictionary, resC, gndLDAC, resLDAmrms, gndLDAmrms,WLDA