from src.langraph.memory_agent import chat

# Turn 1
r1 = chat("test-session", "What were JPMorgan's risks in 2026?")
print("Q1:", r1["query"])
print("Resolved:", r1["resolved_query"])
print("Context preview:", r1["retrieved_context"][:200])
print()

# Turn 2 — follow-up
r2 = chat("test-session", "How do they compare to 2025?")
print("Q2:", r2["query"])
print("Resolved:", r2["resolved_query"])
print("Context preview:", r2["retrieved_context"][:200])

