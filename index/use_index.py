import os
import pyterrier as pt
from sklearn.ensemble import RandomForestRegressor
from pyterrier.measures import Rprec

eval_metrics = ["map", "recip_rank", "ndcg", Rprec]

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
# pt.init()

if not pt.started():
    pt.init(boot_packages=['com.github.terrierteam:terrier-prf:-SNAPSHOT'])

index_dir = '/home/ersel/code/ceng596-termproject/ceng596/index/files/'
index_dir2 = '/home/ersel/code/ceng596-termproject/ceng596/index/files2/'

index = pt.IndexFactory.of(index_dir)
index2 = pt.IndexFactory.of(index_dir2)

bm25 = pt.BatchRetrieve(index, wmodel="BM25")
tfidf = pt.BatchRetrieve(index, wmodel="TF_IDF")
pl2 = pt.BatchRetrieve(index, wmodel="PL2")
dph = pt.BatchRetrieve(index, wmodel="DPH")





models = []
model_names = []
for name in ["BM25", "TF_IDF", "PL2", "DPH"]:
    for ind in [index, index2]:
        models.append(pt.BatchRetrieve(ind, wmodel=name))
        model_names.append(f'{"un" if ind==index2 else ""}stemmed {name}')


exp = pt.Experiment(
    models,
    topics,
    qrels,
    eval_metrics=eval_metrics,
    names=model_names,
)
print(exp)





qe_names = [
    'BA',
    'Bo1',
    'Bo2',
    'Information',
    'KL',
    'KLComplete',
    'KLCorrect',
]
models = []
for qe in qe_names:
    models.append(pt.BatchRetrieve(index, wmodel="BM25", controls={'qe':'on', 'qemodel':qe}))

exp = pt.Experiment(
    models,
    topics,
    qrels,
    eval_metrics=eval_metrics,
    # names=all_model_names,
)
print(exp)






bm25 = pt.BatchRetrieve(index, wmodel="BM25", controls={'qe':'on', 'qemodel':''})


pipeline1 = (tfidf % 200) >> bm25
pipeline2 = (bm25 % 200) >> tfidf
pipeline3 = bm25 >> (tfidf ** pl2)
pipeline4 = pt.FeaturesBatchRetrieve(index, wmodel="TF_IDF", features=["WMODEL:BM25", "WMODEL:PL2"])
# pipeline5 = pt.FeaturesBatchRetrieve(index, wmodel="TF_IDF", features=["WMODEL:BM25", "WMODEL:PL2"])

# topics_train = pt.io.read_topics('AP_collection/topics_train.txt')
# topics_test = pt.io.read_topics('AP_collection/topics_test.txt')
# qrels = pt.io.read_qrels('AP_collection/qrels.txt')

topics = pt.io.read_topics('AP_collection/topics.txt')
qrels = pt.io.read_qrels('AP_collection/qrels.txt')

sdm = pt.rewrite.SequentialDependence()
bo1 = pt.rewrite.Bo1QueryExpansion(index)
rm3 = pt.rewrite.RM3(index)
klq = pt.rewrite.KLQueryExpansion(index)
axq = pt.rewrite.AxiomaticQE(index)

# rf = RandomForestRegressor(n_estimators=400)
# rf_pipe = pipeline5 >> pt.ltr.apply_learned_model(rf)
# rf_pipe.fit(topics_train, qrels)
# pt.Experiment(
#     [bm25, rf_pipe],
#     topics_test,
#     qrels,
#     ["map"],
#     names=["BM25 Baseline", "LTR"]
# )


models = [tfidf, bm25, dph, pl2]
expansion_models = [sdm, bo1, rm3, klq, axq]
all_models = []
model_names = ['tfidf', 'bm25', 'dph', 'pl2']
expansion_names = ['SequentialDependence', 'Bo1QueryExpansion', 'RM3', 'KLQueryExpansion', 'AxiomaticQE']
all_model_names = []
for model,model_name in zip(models,model_names):
    all_models.append(model)
    all_model_names.append(model_name)
    for expansion_model,expansion_name in zip(expansion_models,expansion_names):
        all_models.append(model >> expansion_model >> model)
        all_model_names.append(f'{model_name} WITH {expansion_name}')



exp = pt.Experiment(
    all_models,
    topics,
    qrels,
    eval_metrics=eval_metrics,
    names=all_model_names,
)
print(exp)

exp = pt.Experiment(
    models,
    topics,
    qrels,
    eval_metrics=eval_metrics,
    names=model_names,
)
print(exp)



models = [
    dph >> klq >>dph,
    dph >> bo1 >>dph,
    tfidf >> klq >>tfidf,
    tfidf >> bo1 >>tfidf,
]
model_names = [
    'tfidf WITH Bo1QueryExpansion',
    'tfidf WITH KLQueryExpansion',
    'dph WITH Bo1QueryExpansion',
    'dph WITH KLQueryExpansion',
]
latest_models = []
latest_model_names = []
for model1,name1 in zip(models,model_names):
    for model2,name2 in zip(models,model_names):
        if name1 == name2:
            continue
        latest_models.append((model1%50) >> model2)
        latest_model_names.append(f'{name1} - {name2}')
exp = pt.Experiment(
    latest_models,
    topics,
    qrels,
    eval_metrics=eval_metrics,
    names=latest_model_names,
)
print(exp)









dph = pt.BatchRetrieve(index)
Pipe = dph >> pt.rewrite.stash_results(clear=False) >> pt.rewrite.RM3(index) >> pt.rewrite.reset_results() >> dph

dph >> pt.rewrite.RM3(index) >> dph





query = "prime minister powerful infrastructure"
results = bm25.search(query)




w1 = "Professor"
w2 = "Rafael"
# query = f"#near/3({w1}, {w2})"
tfidf.search(query)

