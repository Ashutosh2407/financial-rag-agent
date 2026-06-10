import asyncio
import os
import csv
from dotenv import load_dotenv
import logging
import time
from datasets import Dataset, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

def batch_helper(dataset,size):
    for i in range(0,len(dataset),size):
        yield dataset[i:i+size]

print("Hello")
dataset_dict = load_dataset("json", data_files="src/eval/datasets/eval_dataset_sparse.json")
dataset = dataset_dict["train"]
for i,item in enumerate(batch_helper(dataset,5)):
    print(f"Starting row {i*5+1}")
    print(item)
    break
    
    