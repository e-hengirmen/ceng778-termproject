import os
import pyterrier as pt
from pyterrier import Terrier

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
# data_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir) + "\\AP_collection\\coll"
# index_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir) + "\\index\\files"

data_dir = 'AP_collection/coll/'
index_dir = './index/files/'
index_dir2 = './index/files2/'

pt.init()
# list of filenames to index
files = pt.io.find_files(data_dir)

# build the index
indexer = pt.TRECCollectionIndexer(
    index_dir,
    verbose=True,
    blocks=True,
    overwrite=True,
)
indexref = indexer.index(files)


indexer = pt.TRECCollectionIndexer(
    index_dir2,
    verbose=True,
    blocks=True,
    overwrite=True,
    stemmer=pt.TerrierStemmer.none,
)
indexref = indexer.index(files)



# load the index, print the statistics
index = pt.IndexFactory.of(indexref)
print(index.getCollectionStatistics().toString())

