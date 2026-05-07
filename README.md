# 🌈 SLAyiNG: A Diverse and Community-validated Dataset of Queer Slang 

[Also available on 🤗](https://huggingface.co/datasets/slaying-neurips-submission/slaying)

---

## Paper

---

## Download the dataset

**Slaying** is available here as a `.json` file. 

Each entry of Slaying contains:

- `id`
- `term`: A queer slang term. Sources for these terms are: GSSO, lgbtDB, Wiktionary, and the Chew Inclusive Terminology Glossary.
- `definition`: Definition for the slang term. Some terms have multiple valid definitions -- for those cases, the definitions are split with bullet points.
- `sentence`: Sentence using the term according to the given definition.
- `communities`: Queer communities associated with the term (via mappings of GSSO and Wiktionary).
- `metadata`: For examples derived from OpenSubtitles, this includes title and IMDb ID. For examples derived from Reddit, this includes the respective subreddit of origin and date. For examples derived from podcasts, this includes the respective title.

---

## Experiments

Partial replication of the paper's experiments is possible with `case_study_i.ipynb` and `case_study_ii.ipynb`. These notebooks require:
- Downloading [QueerReclaimLex](https://github.com/rebedorn/QueerReclaimLex).
- Downloading [WinoQueer](https://github.com/katyfelkner/winoqueer) and following its benchmarking instructions. We test all 1B [DataDecide models](https://huggingface.co/datasets/allenai/DataDecide-data-recipes) accordingly.
- Running the script `run_bpb_datadecide_models.py` on the DataDecide model family to evaluate BPB on **Slaying**.

---

## License
The licenses for the vocabularies sources of **Slaying** are as follows: GSSO falls under Apache License 2.0, lgbtDB under CC0, the Chew Glossary under CC BY-NC-SA 4.0, and Wiktionary under CC-BY-SA. Regarding sentence sources, the OpenSubtitles Corpus falls under ODC-BY. The short excerpts derived from podcast transcripts and Reddit remain the intellectual property of their respective creators and hosting platforms. The derived metadata and annotations produced in this work are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0), where legally permissible. **Slaying** is intended for non-commercial academic research use only.

---

## Misuse
The authors do not condone usage of **Slaying** to enforce or reinforce stereotypes against the LGBTQ+ community. Specifically, terminology or sentences present in **Slaying** should not be used to promote transphobic, homophobic, racist, misogynistic, or otherwise bigoted sentiment and talking points. 
