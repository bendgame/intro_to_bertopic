import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic

df = pd.read_csv('winemag-data-130k-v2.csv')
df.head()

df = df.loc[df.variety == 'Cabernet Sauvignon']
df.describe()

docs = df.description.values

#generate topic model using english defaults
topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
topics, probs = topic_model.fit_transform(docs)
topic_model.get_topic_info()

#cusomizing the topic model
# Step 1 - Extract embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2 - Reduce dimensionality
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

# Step 3 - Cluster reduced embeddings
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# Step 4 - Tokenize topics
vectorizer_model = CountVectorizer(stop_words="english")

# Step 5 - Create topic representation
ctfidf_model = ClassTfidfTransformer()

# All steps together
topic_model = BERTopic(
  embedding_model=embedding_model,    # Step 1 - Extract embeddings
  umap_model=umap_model,              # Step 2 - Reduce dimensionality
  hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
  vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
  ctfidf_model=ctfidf_model,          # Step 5 - Extract topic words
  diversity=0.5,                      # Diversify topic words
  calculate_probabilities=True,        
  verbose=True
)

#using custom embeddings
import tensorflow
import tensorflow_hub as hub

use4 = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

#generate embeddings
use4_embeddings = use4(df['description'])
use= np.array(use4_embeddings)

#create list from np arrays
df['use4'] = use.tolist()

topic_model.fit_transform(docs, use)

#visualize topics
topic_model.visualize_topics()
topic_model.visualize_barchart(top_n_topics=8)
topic_model.visualize_heatmap(n_clusters=20, width=1000, height=1000)

#update the topic model and reduce the topics
topic_model.update_topics(docs, n_gram_range=(1, 2))
topic_model.reduce_topics(docs, nr_topics=6)
