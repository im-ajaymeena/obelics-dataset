import aiohttp
import asyncio
from datasets import Dataset

dataset = Dataset.from_file("obelics-train-00000-of-01439.arrow")

MAX_CONCURRENT_REQUESTS = 500

async def fetch_image(url, session):
    try:
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()
            return await response.read()
    except Exception as e:
        return None

async def download_images(image_urls):
    all_images = []
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def fetch_with_semaphore(url, idx):
            async with semaphore:
                return await fetch_image(url, session), idx

        for urls in image_urls:
            images = [None] * len(urls)
            
            tasks = [fetch_with_semaphore(url, idx) for idx, url in enumerate(urls) if url is not None]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for image_data, idx in results:
                images[idx] = image_data  
            
            all_images.append(images)  
    return all_images

def async_batch_processing(image_urls):
    return asyncio.run(download_images(image_urls))

def update_dataset(batch):
    processed_images = async_batch_processing(batch['images'])
    return {'images': processed_images}

dataset = dataset.map(update_dataset, batched=True, batch_size=100)

dataset.save_to_disk('processed_dataset')
