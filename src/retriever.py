from langchain_community.retrievers import BM25Retriever #community
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever #classic
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereRerank
from dotenv import load_dotenv
import os

load_dotenv()

class Retriever():
    def __init__(self,docs,embeddings,index_name="test-index",k=5):
        self.docs = docs
        self.embeddings = embeddings
        self.index_name = index_name
        self.k = k

    def get_dense_retriever(self):
        vectorstore = PineconeVectorStore(embedding=self.embeddings,index_name=self.index_name)
        retriever = vectorstore.as_retriever(search_kwargs={"k":self.k})
        return retriever

    def get_bm25_sparse_retriever(self):
        retriever = BM25Retriever.from_documents(self.docs)
        retriever.k = self.k
        return retriever
    
    def get_ensemble_retiever(self, weights = [0.4,0.6]):
        retriever = EnsembleRetriever(
            retrievers=[self.get_bm25_sparse_retriver(), self.get_dense_retriver()],
            weights = weights
        )
        return retriever

    def get_contextual_compression_retriever(self, top_n=5, ensemble_weights=[0.4,0.6]):
        top_n = top_n or self.k
        cohere_reranker = CohereRerank(
            cohere_api_key= os.environ["COHERE_API_KEY"],
            model="rerank-english-v3.0",
            top_n=top_n
            )
        
        retriever = ContextualCompressionRetriever(
            base_compressor=cohere_reranker,
            base_retriever= self.get_ensemble_retiever(weights=ensemble_weights)
        )
        return retriever

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever_obj = Retriever(docs=[], embeddings=embeddings)
retriever = retriever_obj.get_dense_retriever()