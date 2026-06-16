from datasets import load_dataset
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import Faithfulness,AnswerRelevancy,ContextPrecision
from openai import AsyncOpenAI
import asyncio
import os
import csv
from dotenv import load_dotenv
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

dataset_dict = load_dataset("json", data_files="src/eval/datasets/eval_dataset_compression.json")
dataset_intermediate = dataset_dict["train"]
dataset = [dict(row) for row in dataset_intermediate]

start = time.time()
#Open AI
openai_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)
print("Key loaded:", os.environ["OPENAI_API_KEY"][:8], flush=True)  # print first 8 chars to verify
ragas_llm = llm_factory(model="gpt-4o-mini", client=openai_client)
logger.info("ragas_llm created.")

#Embeddings
ragas_embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
logger.info("ragas_embedding created.")

faithfulness_score = Faithfulness(llm=ragas_llm)
logger.info("faithfulness_score created.")
answer_relevancy_score = AnswerRelevancy(llm=ragas_llm,embeddings = ragas_embeddings)
logger.info("answer_relevancy_score created.")
context_precision_score = ContextPrecision(llm=ragas_llm)
logger.info("context_precision_score created.")

def batch_helper(dataset,size):
    for i in range(0,len(dataset),size):
        yield dataset[i:i+size]



async def run_ragas(dataset):
    results = []
    logger.info("results=[] created.")

    for i, batch_items in enumerate(batch_helper(dataset,size= 5)):
        logger.info(f"Processing batch {i+1}, questions {i*5+1} to {min((i+1)*5, len(dataset))}")
        for item in batch_items:
            try:
                faith = await asyncio.wait_for(
                        faithfulness_score.ascore(
                        user_input=item["question"],
                        response=item["answer"],
                        retrieved_contexts= "\n\n".join(item["contexts"])
                    )
                    ,timeout=600) 
                logger.info(f"Faithfulness for record {i} calculated... {faith.value}")
            except asyncio.TimeoutError:
                logger.info(f"Q{i} timed out for faithfuless, skipping. Aborting process.")
                continue
            try:
                ars = await asyncio.wait_for(
                        answer_relevancy_score.ascore(
                        user_input=item["question"],
                        response=item["answer"],    
                    )
                    ,timeout=600)
                logger.info(f"Answer relevancy score for record {i} calculated....{ars.value}")
            except asyncio.TimeoutError:
                logger.info(f"Q{i} timed out for answer relevancy, skipping. Aborting process.")
                continue 
            try:
                cps = await asyncio.wait_for(
                        context_precision_score.ascore(
                        user_input= item["question"],
                        retrieved_contexts= item["contexts"],
                        reference= item["ground_truth"],
                    ), 
                    timeout=600
                    )
                logger.info(f"Context precision for record {i} calculated...{cps.value}")
            except asyncio.TimeoutError:
                logger.info(f"Q{i} timed out for context precision, skipping. Aborting process.")
                continue
            results.append({
                    "strategy": item["strategy"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "faithfulness": faith.value,
                    "answer_relevance": ars.value,
                    "context_precision": cps.value,
                })    
        file_exists = os.path.exists("src/eval/results.csv")
        with open("src/eval/results.csv", "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=results[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(results[-len(batch_items):])
        logger.info(f"Batch {i+1} saved.")
        await asyncio.sleep(10) 
    return results

result = asyncio.run(run_ragas(dataset=dataset))
end = time.time()
print(f"Total time required: {end - start:.2f}s")
