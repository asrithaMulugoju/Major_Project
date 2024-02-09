import visualpackage as vis
import bestmapping as bp
import vatmap as vmp
import labelgen as lg
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models
from gensim.sklearn_api import LdaTransformer

def VLDA(n11,tweets,Tn, ntopics):
    tokenizer = RegexpTokenizer(r'\w+')
    texts = []
    for i in range(0,Tn):
        texts.append(tokenizer.tokenize(tweets['t1'][i]))
    #VLDA Model
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    model = LdaTransformer(num_topics=ntopics, id2word=dictionary, iterations=20, random_state=1)
    WLDA= model.fit_transform(corpus)
    RILDA, ODMLDA = vis.vat1(WLDA,1,2)
    mapLDA=vmp.vmap(n11,RILDA,Tn)
    gndLDA = lg.LBGEN(n11, ntopics)
    res2=bp.BestMap(gndLDA,mapLDA)
    return res2, gndLDA, corpus, dictionary