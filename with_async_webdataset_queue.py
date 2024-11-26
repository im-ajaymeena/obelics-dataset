import aiohttp
import asyncio
from datasets import Dataset
import time
from contextlib import contextmanager
import os
import json
import tarfile
from itertools import count
from concurrent.futures import ThreadPoolExecutor
import resource
from multiprocessing import Lock
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

# Dataset setup
dataset = Dataset.from_file("obelics-train-00000-of-01439.arrow")
batch_counter = count(0)

MAX_CONCURRENT_REQUESTS = 2000
MAX_CONNECTIONS = 300
MAX_PROCESS_WORKER = 20

@contextmanager
def profile(section_name):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[PROFILE] {section_name}: {end - start:.4f} seconds")

async def fetch_image(url, session):
    try:
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()
            return await response.read()
    except Exception as e:
        return None

def save_sample_to_tar(row_index, batch_index, shard_index, images, texts, tar_file_path, tar_file_lock):
    """Save a single dataset row directly into a tar file."""
    interleaved = []
    image_counter = 1
    temp_files = []  # Track temporary files to clean up
    img_data_list = []

    for i, (img_data, text) in enumerate(zip(images, texts)):
        if img_data is not None:  # Valid image
            absolute_row_index = batch_index * 1000 + row_index
            image_filename = f"tmp/{shard_index:04}_{absolute_row_index:06d}_{image_counter}.jpg"
            interleaved.append({"image": image_filename})
            temp_files.append(image_filename)
            image_counter += 1
            img_data_list.append((img_data, image_filename))
        elif text is not None:
            interleaved.append({"text": text})
        else:
            return

    for img_data, image_filename in img_data_list:  # Save images
        with open(image_filename, "wb") as img_file:
            img_file.write(img_data)

    metadata_filename = f"tmp/{shard_index:04}_{absolute_row_index:06d}.json"
    with open(metadata_filename, "w") as meta_file:
        json.dump(interleaved, meta_file, ensure_ascii=False, indent=2)
    temp_files.append(metadata_filename)

    with tar_file_lock:
        with tarfile.open(tar_file_path, "a") as tar:
            for file in temp_files:
                tar.add(file, arcname=os.path.basename(file))

async def producer(queue, image_urls, texts, batch_index, progress_bar):
    """Produce download tasks for a single batch."""
    connector = aiohttp.TCPConnector(limit=MAX_CONNECTIONS)
    async with aiohttp.ClientSession(connector=connector) as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def fetch_with_semaphore(url, idx):
            async with semaphore:
                image = await fetch_image(url, session)
                progress_bar.update(1)  # Update progress for each download
                return image, idx

        flattened_urls = [
            (url, row_idx, col_idx)
            for row_idx, row in enumerate(image_urls)
            for col_idx, url in enumerate(row)
            if url is not None
        ]

        tasks = [
            fetch_with_semaphore(url, (row_idx, col_idx))
            for url, row_idx, col_idx in flattened_urls
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_images = [[None] * len(row) for row in image_urls]
        for (image_data, (row_idx, col_idx)) in results:
            all_images[row_idx][col_idx] = image_data

        for row_idx, row in enumerate(all_images):
            await queue.put((row_idx, row, texts[row_idx], batch_index))

async def consumer(queue, tar_file_path, shard_index, batch_count, progress_bar):
    """Consume tasks from the queue and save files."""
    tar_file_lock = Lock()
    completed_batches = 0

    while True:
        task = await queue.get()
        if task is None:  # Sentinel to stop the consumer
            completed_batches += 1
            if completed_batches >= batch_count:
                break
            continue

        row_index, images, texts, batch_index = task
        save_sample_to_tar(row_index, batch_index, shard_index, images, texts, tar_file_path, tar_file_lock)
        progress_bar.update(1)  # Update progress for each save
        queue.task_done()

async def process_all_batches(dataset, batch_size, tar_file_path):
    """Process all batches with a shared queue."""
    queue = asyncio.Queue()
    shard_index = 0
    batch_count = (len(dataset) + batch_size - 1) // batch_size

    # Total tasks for progress bar
    total_downloads = sum(len(batch['images']) * len(batch['images'][0]) for batch in dataset.iter(batch_size=batch_size))
    total_saves = sum(len(batch['images']) for batch in dataset.iter(batch_size=batch_size))

    # Initialize progress bars
    download_bar = tqdm(total=total_downloads, desc="Downloading Images")
    save_bar = tqdm(total=total_saves, desc="Saving Files")

    # Start the consumer
    consumer_task = asyncio.create_task(consumer(queue, tar_file_path, shard_index, batch_count, save_bar))

    # Start producers for each batch
    for batch_index, batch in enumerate(dataset.iter(batch_size=batch_size)):
        image_urls = batch['images']
        texts = batch['texts']
        await producer(queue, image_urls, texts, batch_index, download_bar)

        # Add a sentinel to indicate the producer for this batch is done
        await queue.put(None)

    await consumer_task  # Wait for the consumer to finish
    download_bar.close()
    save_bar.close()

def async_process_all_batches(dataset, batch_size, tar_file_path):
    asyncio.run(process_all_batches(dataset, batch_size, tar_file_path))

def increase_file_descriptor_limit():
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_limit = min(hard_limit, 65536)  # Set a new limit, but do not exceed the hard limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard_limit))
    print(f"File descriptor limit set to: {new_limit}")

# Main execution
increase_file_descriptor_limit()

try:
    os.remove('/home/ajay.meena/obelics/obelics-webdataset/sample.tar')
except OSError:
    pass

async_process_all_batches(dataset, batch_size=1000, tar_file_path='obelics-webdataset/sample.tar')
dataset.save_to_disk('processed_dataset')
