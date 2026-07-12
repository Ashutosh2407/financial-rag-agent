#Metadata filtering

import weaviate
import cohere
import os
from tqdm import tqdm
from weaviate.collections.collection import Collection
from weaviate.classes.query import Filter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

client = weaviate.connect_to_local()
co = cohere.Client(os.environ.get("COHERE_API_KEY"))

collections = {
    "sec_filing": "SECFiling",
    "earnings_transcript": "EarningsTranscript",
    "policy_document": "PolicyDocument"
}

model = SentenceTransformer("all-MiniLM-L6-v2") 

#Cohere Rerank
def rerank(query,results,top_n = 5):
    texts = [obj.properties.get("text","") for obj in results]
    response = co.rerank(
        query =query,
        documents = texts,
        model = "rerank-english-v3.0",
        top_n = top_n
    )
    return [results[hit.index] for hit in response.results]



def _query(collection_name,query,ticker=None,year=None,quarter=None,filing_type=None,limit=5,search_type="dense"):
    
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


    if search_type == "hybrid":
        results =  collection.query.hybrid(
            query=query,
            limit=limit,
            filters=filters,
            alpha = 0.6,
            )   
    else:#dense retrieval
        results =  collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            filters=filters,
            )
    return results.objects
    

def query_all(query,ticker=None,year=None,quarter=None,filing_type=None,limit=5,search_type="dense"):
    
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
        results.extend(hit)
    results = rerank(
        query=query,
        results=results,
        top_n = 5
    )
    return results


"""
r = query_all("What did management say about revenue growth in Q1 2026?",ticker=None,year=None,quarter=None,filing_type=None,limit=5,search_type="dense")
print(r[0])->
GenerativeObject(uuid=_WeaviateUUIDInt('750ba531-f8a6-4368-b50a-9478a86c43f4'), metadata=MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=None, explain_score=None, is_consistent=None, rerank_score=None), properties={'year': 2025, 'page_label': '32', 'total_pages': 77.0, 'source_url': 'https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm', 'creator': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36', 'text': '.” As a result, the Company believes, in general, gross margins will be subject to volatility and downward pressure.\nOperating Expenses\nOperating expenses for 2025, 2024 and 2023 were as follows (dollars in millions):\n2025 Change 2024 Change 2023\nResearch and development $ 34,550\xa0 10\xa0% $ 31,370\xa0 5\xa0% $ 29,915\xa0\nPercentage of total net sales 8% 8% 8%\nSelling, general and administrative $ 27,601\xa0 6\xa0% $ 26,097\xa0 5\xa0% $ 24,932\xa0\nPercentage of total net sales 7% 7% 7%\nTotal operating expenses $ 62,151\xa0 8\xa0% $ 57,467\xa0 5\xa0% $ 54,847\xa0\nPercentage of total net sales 15% 15% 14%\nResearch and Development\nThe growth in R&D expense during 2025 compared to 2024 was primarily driven by increases in headcount-related expenses and\ninfrastructure-related costs.\nSelling, General and Administrative\nThe growth in selling, general and administrative expense during 2025 compared to 2024 was primarily driven by increases in\nheadcount-related expenses and variable selling expenses.\n5/21/26, 7:40 PM aapl-20250927\nhttps://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm 32/77', 'company': 'Apple', 'chunk_id': 'AAPL_10K_2025_p31', 'chunk_index': 1135, 'creationdate': datetime.datetime(2026, 5, 21, 23, 40, 33, tzinfo=datetime.timezone.utc), 'moddate': datetime.datetime(2026, 5, 21, 23, 40, 33, tzinfo=datetime.timezone.utc), 'filing_type': '10-K', 'quarter': '', 'source': '/Users/ashutoshwagh/Desktop/projects/financial-rag-agent/data/apple_10-K.pdf', 'title': 'aapl-20250927', 'page': 31.0, 'producer': 'Skia/PDF m148', 'ticker': 'AAPL'}, references=None, vector={}, collection='SECFiling')
"""

"""
print(results[0].properties)->
{'year': 2025,
 'creator': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36', 
 'total_pages': 77.0, 
 'source_url': 'https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm', 
 'page_label': '32',
    'filing_type': '10-K', 
   'company': 'Apple', 
   'text': '.” As a result, the Company believes, in general, gross margins will be subject to volatility and downward pressure.\nOperating Expenses\nOperating expenses for 2025, 2024 and 2023 were as follows (dollars in millions):\n2025 Change 2024 Change 2023\nResearch and development $ 34,550\xa0 10\xa0% $ 31,370\xa0 5\xa0% $ 29,915\xa0\nPercentage of total net sales 8% 8% 8%\nSelling, general and administrative $ 27,601\xa0 6\xa0% $ 26,097\xa0 5\xa0% $ 24,932\xa0\nPercentage of total net sales 7% 7% 7%\nTotal operating expenses $ 62,151\xa0 8\xa0% $ 57,467\xa05\xa0% $ 54,847\xa0\nPercentage of total net sales 15% 15% 14%\nResearch and Development\nThe growth in R&D expense during 2025 compared to 2024 was primarily driven by increases in headcount-relatedexpenses and\ninfrastructure-related costs.\nSelling, General and Administrative\nThe growth in selling, general and administrative expense during 2025 compared to 2024 was primarily driven by increases in\nheadcount-related expenses and variable selling expenses.\n5/21/26, 7:40 PM aapl-20250927\nhttps://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm 32/77', 
   'chunk_index': 1135, 
   'creationdate': datetime.datetime(2026, 5, 21, 23, 40, 33,
     tzinfo=datetime.timezone.utc), 
     'moddate': datetime.datetime(2026, 5, 21, 23, 40, 33, tzinfo=datetime.timezone.utc), 
     'source': '/Users/ashutoshwagh/Desktop/projects/financial-rag-agent/data/apple_10-K.pdf', 
     'quarter': '', 'chunk_id': 'AAPL_10K_2025_p31', 
     'producer': 'Skia/PDF m148', 'page': 31.0, 
     'title': 'aapl-20250927', 
     'ticker': 'AAPL'}"""
