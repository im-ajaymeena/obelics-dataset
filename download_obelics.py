from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceM4/OBELICS",
    cache_dir="/home/ajay.meena/obelics/huggiface_cache",
    num_proc=40,
)
