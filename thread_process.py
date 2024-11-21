from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from PIL import Image
import requests

from datasets import Dataset
dataset = Dataset.from_file("obelics-train-00000-of-01439.arrow")

# Define the number of threads and processes based on your system's capabilities
MAX_DOWNLOAD_THREADS = 120

# Function to fetch a single image
def fetch_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except Exception as e:
        # logging.error(f"Error fetching image: {e}")
        return None

# Function to download all images using threads
def download_images(image_urls):
    all_images = []  # To store the results as a list of lists
    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_THREADS) as executor:
        # Process each sublist (batch) of URLs
        for urls in image_urls:
            # Filter out None values and create a mapping for valid URLs
            valid_idx_url_pairs = [(idx, url) for idx, url in enumerate(urls) if url is not None]
            futures = {executor.submit(fetch_image, url): idx for idx, url in valid_idx_url_pairs}
            
            # Initialize the placeholder list with None values for the current batch
            images = [None] * len(urls)
            
            # Collect the results
            for future in as_completed(futures):
                idx = futures[future]
                images[idx] = future.result() if future.exception() is None else None
            
            all_images.append(images)  # Append the processed batch to the main list
    return all_images

# Apply processing with the map function using batched processing
def update_dataset(batch):
    processed_batch = download_images(batch['images'])
    return {'images': processed_batch}

# Assuming you have a dataset loaded as `dataset`
dataset = dataset.map(update_dataset, batched=True, batch_size=200)

# Save the processed dataset to disk
dataset.save_to_disk('processed_dataset')
