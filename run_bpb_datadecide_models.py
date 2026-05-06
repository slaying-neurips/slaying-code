import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cache_dir = None # add cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ["HF_TOKEN"] = None # add HF_TOKEN

import torch
import json
import pickle
import math
from tqdm import tqdm
from transformers import AutoTokenizer
from hf_olmo import OLMoForCausalLM  # pip install ai2-olmo


with open('slaying.json', 'r') as f:
    data = json.load(f)

COMMUNITY_MODE = True  # whether to do community-based analysis

# ── Toggle ────────────────────────────────────────────────────────────────────
# "recipes" → fix size at 1B, sweep all data recipes
# "sizes"   → fix recipe at FIXED_RECIPE, sweep all model sizes
SWEEP_MODE = "recipes"   # "recipes" | "sizes"

# ── DataDecide config ─────────────────────────────────────────────────────────
ALL_RECIPES: list[str] = [
    "dolma1_7-1B",
    "dolma1_7-no-code-1B",
    "dolma1_7-no-math-code-1B",
    "dolma1_7-no-reddit-1B",
    "dolma1_7-no-flan-1B",
    "dolma1_6plus-1B",
    "c4-1B",
    "fineweb-pro-1B",
    "fineweb-edu-1B",
    "falcon-1B",
    "falcon-and-cc-1B",
    "falcon-and-cc-qc-10p-1B",
    "falcon-and-cc-qc-20p-1B",
    "falcon-and-cc-qc-orig-10p-1B",
    "falcon-and-cc-qc-tulu-10p-1B",
    "dclm-baseline-1B",
    "dclm-baseline-qc-7p-fw2-1B",
    "dclm-baseline-qc-7p-fw3-1B",
    "dclm-baseline-qc-fw-3p-1B",
    "dclm-baseline-qc-fw-10p-1B",
    "dclm-baseline-qc-10p-1B",
    "dclm-baseline-qc-20p-1B",
    "dclm-baseline-25p-dolma1.7-75p-1B",
    "dclm-baseline-50p-dolma1.7-50p-1B",
    "dclm-baseline-75p-dolma1.7-25p-1B",
]
print(f"Recipes: {len(ALL_RECIPES)}")

ALL_SIZES: list[str] = [
    "4M", "6M", "8M", "10M", "14M", "16M", "20M",
    "60M", "90M", "150M", "300M", "530M", "750M", "1B",
]

# Fixed recipe used when SWEEP_MODE = "sizes"
FIXED_RECIPE = "dolma1_7"

VARIANTS = ["sentence"] 

device = "cuda" if torch.cuda.is_available() else "cpu"

### save community dictionary

# community-based analysis

with open('gsso.json', 'r') as f:
    gsso = json.load(f)

with open('wiktionary_transgender_results.json', 'r') as f:
    wiki_transgender = json.load(f)
    wiki_transgender_terms = list(wiki_transgender.keys())

with open('wiktionary_gay_results.json', 'r') as f:
    wiki_gay = json.load(f)
    wiki_gay_terms = list(wiki_gay.keys())

with open('wiktionary_drag_results.json', 'r') as f:
    wiki_drag = json.load(f)
    wiki_drag_terms = list(wiki_drag.keys())

term_community = {}
for entry in data:
    term = entry['term']
    if term in term_community:
        continue 
    term_community[term] = []
    for community_slang, examples in gsso.items():
        community = community_slang.replace("slang", "").strip()
        for example in examples:
            if example['label'].lower() == term or term in example['synonyms']:
                term_community[term].append(community)
            if term in wiki_transgender_terms and "transgender" not in term_community[term]:
                term_community[term].append('transgender')
            if term in wiki_gay_terms and "gay" not in term_community[term]:
                term_community[term].append('gay')
            if term in wiki_drag_terms and "drag" not in term_community[term]:
                term_community[term].append('drag')

def compute_ppl(text: str, model, tokenizer) -> float:
    """
    Standard token-normalised perplexity:
        PPL = exp( mean NLL per token )
    which is exactly what HuggingFace returns as model.loss when labels=input_ids.
    """
    enc = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        loss = model(enc["input_ids"], labels=enc["input_ids"]).loss
    return math.exp(loss.item())


# ── BPB ───────────────────────────────────────────────────────────────────────
def compute_bpb(text: str, model, tokenizer) -> float:
    n_bytes = len(text.encode("utf-8"))
    enc = tokenizer(text, return_tensors="pt").to(device)
    n_tokens = enc["input_ids"].shape[1]
    with torch.no_grad():
        loss = model(enc["input_ids"], labels=enc["input_ids"]).loss
    return (loss.item() * n_tokens) / (math.log(2) * n_bytes)


# ── Build model list depending on mode ───────────────────────────────────────
if SWEEP_MODE == "recipes":
    model_list: list[tuple[str, str]] = [
        (recipe, f"allenai/DataDecide-{recipe}")
        for recipe in ALL_RECIPES
    ]
    output_pkl = "datadecide_recipes_final_bpb.pkl"

elif SWEEP_MODE == "sizes":
    model_list = [
        (size, f"allenai/DataDecide-{FIXED_RECIPE}-{size}")
        for size in ALL_SIZES
    ]
    output_pkl = f"datadecide_{FIXED_RECIPE}_bpb_sizes.pkl"

else:
    raise ValueError(f"Unknown SWEEP_MODE: {SWEEP_MODE!r}")

if COMMUNITY_MODE:
    output_pkl = output_pkl.replace(".pkl", "_communities.pkl")


# ── Sweep ─────────────────────────────────────────────────────────────────────
results: dict[str, dict[str, list[float]]] = {}

for label, model_id in model_list:
    print(f"\n{'─'*60}\nLoading: {model_id}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = OLMoForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
        model.eval().to(device)
    except Exception as e:
        print(f"  !! Failed to load {model_id}: {e}")
        continue

    results[label] = {v: [] for v in VARIANTS}

    for entry in tqdm(data, desc=label):

        for v in VARIANTS:
            bpb = compute_bpb(entry[v], model, tokenizer)

            # community
            if COMMUNITY_MODE:
                for term, communities in term_community.items():
                    if term == entry['term']:
                        coms = communities

                results[label][v].append((bpb, coms, entry['term']))
            else:
                results[label][v].append(bpb)


    with open(output_pkl, 'wb') as f:
        pickle.dump(results, f)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\nDone. Results saved to {output_pkl}")