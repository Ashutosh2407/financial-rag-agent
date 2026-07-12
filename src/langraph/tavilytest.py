import time,os
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

tool = TavilySearchResults(
    api_key=os.environ.get("TAVILY_API_KEY"),
    max_results=5,
    include_answer=True,
    include_raw_content=False,
)

start = time.time()
results = tool.invoke({"query": "When is apple iphone product launch"})
print(f"Took {time.time() - start:.2f}s")