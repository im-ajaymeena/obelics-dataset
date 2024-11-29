import aiohttp
import asyncio
from datasets import Dataset
import time
from contextlib import contextmanager
import os, json
from itertools import count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import argparse
import tarfile
from tqdm import tqdm

@contextmanager
def profile(section_name):
    if args.disable_profiling:
        yield
        return
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

shard_save_dir = None

import aiohttp
import asyncio

MAX_CONCURRENT_REQUESTS = 2000
MAX_CONNECTIONS = 300
MAX_PROCESS_WORKER = 40

def save_sample_to_tar(row_index, batch_index, images, texts):
    """Save a single dataset row directly into a tar file."""
    # try:
    # Interleaved metadata creation
    interleaved = []
    image_counter = 1
    temp_files = []  # Keep track of temporary files to clean up
    img_data_list = []

    output_dir = f"{shard_save_dir}/{batch_index:04}"
    os.makedirs(output_dir, exist_ok=True)

    for i, (img_data, text) in enumerate(zip(images, texts)):
        if img_data is not None:  # Valid image
            absolute_row_index = batch_index*1000 + row_index
            image_filename = f"{output_dir}/{absolute_row_index:06d}_{image_counter}.jpg"
            interleaved.append({"image": image_filename})
            temp_files.append(image_filename)
            image_counter += 1
            img_data_list.append((img_data, image_filename))
        elif text is not None:
            interleaved.append({"text": text})
        else:
            return
        
    for img_data, image_filename in img_data_list:  # Unpack and save each image
        with open(image_filename, "wb") as img_file:
            img_file.write(img_data)

    # Save metadata to a temporary JSON file
    metadata_filename = f"{output_dir}/{absolute_row_index:06d}.json"
    with open(metadata_filename, "w") as meta_file:
        json.dump(interleaved, meta_file, ensure_ascii=False, indent=2)
    
    return

async def save_batch_with_multiprocessing(loop, batch_index, images, texts):
    """Process a batch using multiprocessing within asyncio."""
    with ThreadPoolExecutor(max_workers=MAX_PROCESS_WORKER) as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                save_sample_to_tar,
                row_index,
                batch_index,
                images[row_index],
                texts[row_index],
            )
            for row_index in range(len(images))
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        return

async def download_images(image_urls, texts, batch_index):
    # headers = {"Accept-Encoding": "gzip, deflate, br"}
    connector = aiohttp.TCPConnector(limit=MAX_CONNECTIONS)
    async with aiohttp.ClientSession(connector=connector) as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def fetch_with_semaphore(url, idx):
            async with semaphore:
                image = await fetch_image(url, session)
                return image, idx
            
        with profile("Pre process"):
            # Flatten URLs while preserving structure
            flattened_urls = [
                (url, row_idx, col_idx)
                for row_idx, row in enumerate(image_urls)
                for col_idx, url in enumerate(row)
                if url is not None
            ]

            # Create tasks for all valid URLs
            tasks = [
                fetch_with_semaphore(url, (row_idx, col_idx))
                for url, row_idx, col_idx in flattened_urls
            ]

        # Execute all tasks with a single gather
        with profile("Async gather"):
            results = await asyncio.gather(*tasks, return_exceptions=True)


        with profile("Post process"):
            # Initialize the result structure with None
            all_images = [[None] * len(row) for row in image_urls]

            # Fill the results back into the original structure
            for (image_data, (row_idx, col_idx)) in results:
                all_images[row_idx][col_idx] = image_data
        
    with profile("Save to webdataset"):
        loop = asyncio.get_running_loop()
        await save_batch_with_multiprocessing(loop, batch_index, all_images, texts)    
    return all_images

batch_counter = count(0)

def async_batch_processing(image_urls, text_list, batch_index):
    result = asyncio.run(download_images(image_urls, text_list, batch_index))
    return result


def update_dataset(batch):
    batch_index = next(batch_counter)
    processed_images = async_batch_processing(batch['images'], batch['texts'], batch_index)
    return {'images': processed_images}

import resource

def increase_file_descriptor_limit():
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_limit = min(hard_limit, 65536)  # Set a new limit, but do not exceed the hard limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard_limit))
    print(f"File descriptor limit set to: {new_limit}")

increase_file_descriptor_limit()

def create_flat_tar(root_dir, tar_filename):
    all_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
    
    with tarfile.open(tar_filename, "w") as tar:
        for file in tqdm(all_files):
            tar.add(file, arcname=os.path.basename(file))

    print(f"Created tar file: {tar_filename}")
    print(f"Total files added: {len(all_files)}")


parser = argparse.ArgumentParser(description="Process an arrow file.")
parser.add_argument("--arrow_file", type=str, required=False, default="obelics-train-00002-of-01439.arrow", help="Path to the .arrow file")
parser.add_argument("--disable_profiling", action="store_true", help="Disable profiling for the script")
args = parser.parse_args()

arrow_file = args.arrow_file
shard_index = arrow_file.split('-')[2]
shard_save_dir = f"obelics-object-storage/{shard_index:05}"
os.makedirs(shard_save_dir, exist_ok=True)

dataset = Dataset.from_file(arrow_file)

dataset = dataset.map(update_dataset, batched=True, batch_size=1000)

tarfile_path = f"obelics-web-dataset/{shard_index:05}.tar"
os.makedirs(os.path.dirname(tarfile_path), exist_ok=True)
create_flat_tar(shard_save_dir, tarfile_path)