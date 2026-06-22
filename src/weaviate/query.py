#Metadata filtering

import weaviate
from tqdm import tqdm
from weaviate.collections.collection import Collection
from weaviate.classes.query import Filter
from sentence_transformers import SentenceTransformer

client = weaviate.connect_to_local()

collections = {
    "sec_filing": "SECFiling",
    "earnings_transcript": "EarningsTranscript",
    "policy_document": "PolicyDocument"
}

model = SentenceTransformer("all-MiniLM-L6-v2") 

def _query(collection_name,query,ticker=None,year=None,quarter=None,filing_type=None,limit=5):
    
    collection = client.collections.get(collection_name)
    query_vector = model.encode(query).tolist()
    filters = None

    if ticker:
        filters = Filter.by_property("ticker").equal(ticker)
    if year:
        filter_year = Filter.by_property("year").equal(year)
        filters = filters & filter_year if filters else filter_year
    if quarter:
        filter_quarter = Filter.by_property("quarter").equal(quarter)
        filters = filters & filter_quarter if filters else filter_quarter
    if filing_type:
        filter_filing_type = Filter.by_property("filing_type").equal(filing_type)
        filters = filters & filter_filing_type if filters else filter_filing_type


    return collection.query.near_vector(near_vector=query_vector,limit=limit,filters=filters)

def query_all(query,ticker=None,year=None,quarter=None,filing_type=None,limit=5):
    
    results = []
    for collection_name in collections.values():
        hit = _query(collection_name=collection_name,
                        query = query,
                        ticker=ticker, 
                        year=year,
                        quarter=quarter,
                        filing_type=filing_type,
                        limit=limit
                        )
        results.extend(hit.objects)
    
    return results


