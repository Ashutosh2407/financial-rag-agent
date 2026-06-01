from src.chunking import get_chunking_context
from langchain_community.document_loaders import PyPDFLoader


def load_docs(path: str):
    loader = PyPDFLoader(path)
    doc = loader.load()
    print(f"{path.split("/")[-1]}: {len(doc)} pages.")
    return doc



if __name__ == "__main__":
    doc = load_docs("/Users/ashutoshwagh/Desktop/projects/financial-rag-agent/data/jpm-20251231.pdf")
    ctx = get_chunking_context("fixed")
    #print(type(doc[0]))
    chunks = ctx.chunk(doc)
    
    
    
    #Open the file in write mode ('w')
    with open("fixed.txt", "w") as file:
        file.write(f"\nTotal chunks: {len(chunks)}\n")
        for i in range(len(chunks)):
            file.write(f"\n--- Chunk {i} ---\n{chunks[i].page_content}")

    
    

