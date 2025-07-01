# Automating-Clinical-Trial-Matches-via-NLP-of-Synthetic-EHRs-and-Trial-EligibilityCriteria-
TrialMatcher was initially prototyped using the publicly available Synthetic Suicide Prevention (SSP) Dataset with 
Social Determinants of Health17 published by the U.S. Department of Veterans Affairs. This dataset contained 10,000 
synthetically generated Veteran patient records made with the platform Synthea and encompassed over 500 clinical 
concepts. This dataset is comprised of numerical and textual information regarding patient allergies, procedures, 
conditions, devices, immunizations, demographics, lab results, medications, and insurance data.  
The workflow applied to the SSP dataset is shown in Figure 1. The entire set of 10,000 patients comprises the 
population data in this case. The data was transformed into wide format since the information for each patient was 
spread across multiple rows and then all dataframes comprising the dataset were merged using the unique patient ID. 
This resulted in data being placed in lists that contained all allergies, medications, conditions, etc. found in the patient 
profile. Then, the full patient profile for any patient in the dataset could be readily extracted. Next, the textual 
descriptions for the patient were preprocessed through a spacy pipeline involving tokenization, part of speech tagging, 
a sentence recognizer, a dependency parser, a lemmatizer, an abbreviation detector for biomedical text from the 
scispacy library and finalizing with an NER component that used one of the models shown in Table 1. The extracted 
entities from the NER can then be linked to various clinical concepts via NEL through the UMLS API which are 
important as they could enable a developer or a patient to only query for clinical trials that deal with conditions, 
medications, procedures, etc. or a combination thereof.  
The extracted entities serve to define the medical profile of a patient and each term can be used to query for clinical 
trials via the clinicaltrials.gov API. Through this API, one can generate the same results as one would through their 
browser implementation that can be saved as a .csv file and subsequently loaded into a pandas dataframe. This means 
that any filters concerning age, sex, location, study phase and recruitment status can be readily applied to the query to 
refine the results. The application of these filters is important since it is a straightforward way to narrow down the 
number of trials that will need to be parsed which can significantly speed up the time it takes to provide results. 
Included in the clinical trial query output are textual descriptions of the eligibility criteria for each trial. Eligibility is 
defined by a set of inclusion and exclusion criteria that appear in a standardized format, hence, regular expressions 
can be used to split each trial’s matching criteria into its respective components. Then, both inclusion and exclusion 
criteria can undergo through the same NLP pipeline described earlier. The result of this process generates a set of 
concepts that define the patient profile and a set of concepts for each clinical trial whose similarity can be quantified 
using similarity metrics like the Sørensen-Dice Index (SDI) 
