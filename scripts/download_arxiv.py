from huggingface_hub import hf_hub_download
import json

DATA_DIR = "./data/arxiv"

# List of all files in the dataset  
all_files = [
    "database_vectors.fvecs",
    "query_vectors.fvecs",
    "database_attributes.jsonl",
    "em_query_attributes.jsonl",
    "r_query_attributes.jsonl",
    "emis_query_attributes.jsonl",
    "ground_truth_em.ivecs",
    "ground_truth_r.ivecs",
    "ground_truth_emis.ivecs"
]

# Mapping of file names to their respective paths after downloading
mapping_of_filenames_to_paths = {}
 
# Iterate over all files and download each one 
for file in all_files:
    path = hf_hub_download(
        "SPCL/arxiv-for-fanns-large",
        filename=file,
        repo_type="dataset",
        local_dir=DATA_DIR,
        cache_dir=None
    )
    mapping_of_filenames_to_paths[file] = path
print(mapping_of_filenames_to_paths)

def get_labels(filename):
    # First pass: collect all unique main categories
    all_categories = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            main_cats = entry.get("main_categories", [])
            all_categories.update(main_cats)
    return all_categories

all_categories = get_labels(mapping_of_filenames_to_paths["database_attributes.jsonl"])
category_to_id = {cat: i for i, cat in enumerate(sorted(all_categories))}

def write_labels(input_file, output_file, category_to_id):
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            entry = json.loads(line)
            main_cats = entry.get("main_categories", [])
            ids = sorted([category_to_id[cat] for cat in main_cats if cat in category_to_id])
            fout.write(",".join(str(id) for id in ids) + "\n")

write_labels(mapping_of_filenames_to_paths["database_attributes.jsonl"],
    DATA_DIR + "/labels_diskann.txt",category_to_id)

def write_labels_q(input_file, output_file, category_to_id):
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            entry = json.loads(line)
            main_cats = [entry.get("label", "??")]
            for cat in main_cats:
                assert cat in category_to_id
            ids = sorted([category_to_id[cat] for cat in main_cats if cat in category_to_id])
            fout.write(",".join(str(id) for id in ids) + "\n")

write_labels_q(mapping_of_filenames_to_paths["emis_query_attributes.jsonl"],
    DATA_DIR + "/query_labels_diskann.txt",category_to_id)
