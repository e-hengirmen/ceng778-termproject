import os
import pyterrier as pt


os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
pt.init()


index_dir = './index/files/'
index = pt.IndexFactory.of(index_dir)

# bm25 = pt.BatchRetrieve(index, wmodel="BM25")
tfidf = pt.BatchRetrieve(index, wmodel="TF_IDF")
# pl2 = pt.BatchRetrieve(index, wmodel="PL2")
dph = pt.BatchRetrieve(index, wmodel="DPH")



sdm = pt.rewrite.SequentialDependence()
# bo1 = pt.rewrite.Bo1QueryExpansion(index)
# rm3 = pt.rewrite.RM3(index)
klq = pt.rewrite.KLQueryExpansion(index)
# axq = pt.rewrite.AxiomaticQE(index)

pipeline = (dph >> klq >>dph) % 20

def search_query(query):
    # results = pipeline.search(query)
    results = pipeline.search(query)
    print(results)
    best_20 = [i for i in results[:20]['docno']]
    return best_20




