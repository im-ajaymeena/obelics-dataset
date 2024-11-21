import aiohttp
import asyncio
from datasets import Dataset
import time
from contextlib import contextmanager

dataset = Dataset.from_file("obelics-train-00000-of-01439.arrow")


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

import aiohttp
import asyncio

MAX_CONCURRENT_REQUESTS = 2000
MAX_CONNECTIONS = 300

async def download_images(image_urls):
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

        print(len(tasks), 'task-len')

        # Execute all tasks with a single gather
        with profile("Async gather"):
            results = await asyncio.gather(*tasks, return_exceptions=True)


        with profile("Post process"):
            # Initialize the result structure with None
            all_images = [[None] * len(row) for row in image_urls]

            # Fill the results back into the original structure
            for (image_data, (row_idx, col_idx)) in results:
                all_images[row_idx][col_idx] = image_data
        

    return all_images


def async_batch_processing(image_urls):
    result = asyncio.run(download_images(image_urls))
    return result

def update_dataset(batch):
    processed_images = async_batch_processing(batch['images'])
    return {'images': processed_images}

dataset = dataset.map(update_dataset, batched=True, batch_size=1000)

dataset.save_to_disk('processed_dataset')
