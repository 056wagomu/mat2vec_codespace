from gensim.models import Word2Vec
from mat2vec.processing import MaterialsTextProcessor

w2v_model = Word2Vec.load(".venv/lib/python3.7/site-packages/mat2vec/training/models/pretrained_embeddings")
text_processor = MaterialsTextProcessor()
answer = w2v_model.wv.most_similar("selenium", topn=5)

print(answer)