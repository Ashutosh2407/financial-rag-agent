from langchain_community.retrievers import BM25Retriever #community
from langchain_classic.retrievers import EnsembleRetriever #classic
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

class Retriever():
    def __init__(self,docs,embeddings,index_name="test-index",k=5):
        self.docs = docs
        self.embeddings = embeddings
        self.index_name = index_name
        self.k = k

    def get_dense_retriver(self):
        vectorstore = PineconeVectorStore(embedding=self.embeddings,index_name=self.index_name)
        retriever = vectorstore.as_retriever(search_kwargs={"k":self.k})
        return retriever

    def get_bm25_sparse_retriver(self):
        retriever = BM25Retriever.from_documents(self.docs)
        retriever.k = self.k
        return retriever
    
    def get_ensemble_retiever(self, weights = [0.4,0.6]):
        retriver = EnsembleRetriever(
            retrievers=[self.get_bm25_sparse_retriver(), self.get_dense_retriver()],
            weights = weights
        )
        return retriver


