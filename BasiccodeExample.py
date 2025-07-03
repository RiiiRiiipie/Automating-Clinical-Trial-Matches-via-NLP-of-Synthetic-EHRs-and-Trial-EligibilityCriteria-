# TrialMatcher: Full NLP Pipeline for Patient-Trial Matching using BioSyn, UMLS & ClinicalTrials.gov

# -------------------------
# STEP 0: Import Libraries & Setup
# -------------------------

import pandas as pd
import requests
import re
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from biosyn import BioSyn

# -------------------------
# STEP 1: Load Sample Patient EHR (Replace with real EHR text)
# -------------------------
ehr_text = """
The 58-year-old male patient has a diagnosis of type 2 diabetes mellitus,
and has been prescribed metformin (500mg/day). The patient also has a
history of hypertension but no prior incidents of myocardial infarction.
"""

# -------------------------
# STEP 2: Preprocess EHR (optional but recommended)
# -------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9., ]", "", text)
    return text.strip()

ehr_text_clean = preprocess_text(ehr_text)
print("[Cleaned EHR Text]:\n", ehr_text_clean)

# -------------------------
# STEP 3: Named Entity Recognition (NER)
# -------------------------
ner_tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
ner_model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")
ner_pipe = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")

ner_results = ner_pipe(ehr_text_clean)
ehr_mentions = list(set([ent['word'] for ent in ner_results]))
print("\n[NER] Extracted Mentions:", ehr_mentions)

# -------------------------
# STEP 4: BioSyn Entity Linking (UMLS Mapping)
# -------------------------
biosyn = BioSyn()
biosyn.load_model("dmis-lab/biosyn-sapbert-umls", output_dir="outputs", device="cpu")
biosyn.load_dictionary("umls_dictionary.tsv")  # format: CUI <TAB> Name

ehr_linked = []
for mention in ehr_mentions:
    try:
        candidate = biosyn.retrieve_candidate(mention, topk=1)[0]
        ehr_linked.append({
            "mention": mention,
            "umls_name": candidate['name'],
            "cui": candidate['cui'],
            "score": candidate['score']
        })
    except:
        continue

print("\n[UMLS Mapped Concepts]:")
for concept in ehr_linked:
    print(f"{concept['mention']} -> {concept['umls_name']} ({concept['cui']})")

# -------------------------
# STEP 5: Query clinicaltrials.gov API
# -------------------------
query_terms = " AND ".join([c['umls_name'] for c in ehr_linked])
api_url = f"https://clinicaltrials.gov/api/query/full_studies?expr={query_terms}&min_rnk=1&max_rnk=5&fmt=json"
response = requests.get(api_url).json()

# -------------------------
# STEP 6: Extract Eligibility Criteria
# -------------------------
trials = []
for study in response.get("FullStudiesResponse", {}).get("FullStudies", []):
    try:
        ident = study["Study"]["ProtocolSection"]["IdentificationModule"]
        elig = study["Study"]["ProtocolSection"]["EligibilityModule"]
        trials.append({
            "NCTId": ident.get("NCTId", ""),
            "Title": ident.get("OfficialTitle", ""),
            "Eligibility": elig.get("EligibilityCriteria", "")
        })
    except KeyError:
        continue

print(f"\n[Fetched {len(trials)} Matching Trials]")

df = pd.DataFrame(trials)
df.to_csv("matched_clinical_trials.csv", index=False)

# -------------------------
# STEP 7: NER on Trial Eligibility (Optional Deep Matching)
# -------------------------
trial_matches = []
for trial in trials:
    trial_text = preprocess_text(trial['Eligibility'])
    trial_ner = ner_pipe(trial_text)
    trial_mentions = list(set([ent['word'] for ent in trial_ner]))

    trial_linked = []
    for mention in trial_mentions:
        try:
            candidate = biosyn.retrieve_candidate(mention, topk=1)[0]
            trial_linked.append(candidate['cui'])
        except:
            continue

    # Compare patient CUIs vs trial CUIs using SDI
    patient_cuis = set([c['cui'] for c in ehr_linked])
    trial_cuis = set(trial_linked)

    intersection = len(patient_cuis & trial_cuis)
    union = len(patient_cuis) + len(trial_cuis)
    sdi = 2 * intersection / union if union else 0

    trial_matches.append({
        "NCTId": trial['NCTId'],
        "Title": trial['Title'],
        "SDI Match Score": round(sdi, 2)
    })

print("\n[Trial Matching Based on SDI Score]:")
for match in trial_matches:
    print(match)

# -------------------------
# STEP 8: Export Final Results
# -------------------------
final_df = pd.DataFrame(trial_matches)
final_df.to_csv("final_trial_match_scores.csv", index=False)
print("\n✅ Saved SDI-based matching scores to final_trial_match_scores.csv")
