from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
import logging
from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
import concurrent.futures
from tqdm import tqdm
import logging
from datasets import Dataset


from datasets import load_dataset
from tqdm import tqdm


dataset = Dataset.from_file("obelics-train-00000-of-01439.arrow")



import datasets
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import requests
from io import BytesIO
import tqdm

# Load your dataset

# Define the number of threads based on your system's capabilities
MAX_WORKERS = 50

# Function to fetch a single image
def fetch_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        
        # Convert to RGBA if the image has a transparency channel, otherwise use RGB
        if img.mode in ('RGBA', 'LA') or ('transparency' in img.info):
            img = img.convert('RGBA')
        else:
            img = img.convert('RGB')
        
        return img
    except Exception as e:
        print(f"Error fetching image: {e}")
        return None



# Function to process a list of URLs
def process_image_list(image_list):
    idx_url_pairs = [(idx, url) for idx, url in enumerate(image_list) if url is not None]
    results = [None] * len(image_list)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {executor.submit(fetch_image, url): idx for idx, url in idx_url_pairs}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result() if future.exception() is None else None
    return results

# Function to process a batch of data
def process_batch(batch):
    batch_size = len(batch['images'])
    processed_images = [process_image_list(images) for images in batch['images']]
    return {'images': processed_images}

# Apply the processing using the map function with batched processing
def update_dataset(batch):
    processed_batch = process_batch(batch)
    return {'images': processed_batch['images']}

dataset = dataset.map(update_dataset, batched=True, batch_size=100)

# Save the processed dataset to disk
dataset.save_to_disk('processed_dataset')
