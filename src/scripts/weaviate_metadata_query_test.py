from src.weaviate.query import query_all

#Query1: General Question. No metadata
print("=== Test 1: No filters ===")
results = query_all(
    query="What did management say about revenue growth in Q1 2026?"
)
for r in results:
    print("Test 1")
    print(f"Query:What did management say about revenue growth in Q1 2026?")
    print(r.properties.get("ticker"), r.properties.get("filing_type"), r.properties.get("year"))
    print(r.properties.get("text")[:150])
    print("---")

#Query2: General Question. Ticker mention
print("=== Test 2: Ticker ===")
results = query_all(
    query="What did management say about revenue growth in Q1 2026?",
    ticker = "AAPL"
)
for r in results:
    print("Test 2")
    print(f"Query:What did management say about revenue growth in Q1 2026?")
    print(r.properties.get("ticker"), r.properties.get("filing_type"), r.properties.get("year"))
    print(r.properties.get("text")[:150])
    print("---")

#Query3: General Question. Ticker and year mention.
print("=== Test 3: Ticker and Year ===")
results = query_all(
    query="What did management say about revenue growth in Q1 2026?",
    ticker = "AAPL",
    year=2026
)
for r in results:
    print("Test 3")
    print(f"Query:What did management say about revenue growth in Q1 2026?")
    print(r.properties.get("ticker"), r.properties.get("filing_type"), r.properties.get("year"))
    print(r.properties.get("text")[:150])
    print("---")

#Query4: General Question. Ticker,year and quarter mention.
print("=== Test 4: Ticker,Year and quarter ===")
results = query_all(
    query="What did management say about revenue growth in Q1 2026?",
    ticker = "AAPL",
    year=2026,
    quarter="Q1"
)
for r in results:
    print("Test 4")
    print(f"Query:What did management say about revenue growth in Q1 2026?")
    print(r.properties.get("ticker"), r.properties.get("filing_type"), r.properties.get("year"), r.properties.get("quarter"))
    print(r.properties.get("text")[:150])
    print("---")

#Query5: General Question. Ticker,year, quarter and filing type mention.
print("=== Test 5: Ticker,Year, quarter and filing type===")
results = query_all(
    query="What did management say about revenue growth in Q1 2026?",
    ticker = "AAPL",
    year=2026,
    quarter="Q1",
    filing_type="Earnings Call Transcript"
)
for r in results:
    print("Test 5")
    print(f"Query:What did management say about revenue growth in Q1 2026?")
    print(r.properties.get("ticker"), r.properties.get("filing_type"), r.properties.get("year"), r.properties.get("quarter"))
    print(r.properties.get("text")[:150])
    print("---")