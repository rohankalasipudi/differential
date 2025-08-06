import json
import spacy
import streamlit as st
from collections import defaultdict

# Load the NLP model
nlp = spacy.load("en_core_web_sm")

# Load symptoms data
def load_symptoms(file_path="symptoms.json"):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data["symptoms"]
    except FileNotFoundError:
        st.error("Error: symptoms.json file not found.")
        return []
    except json.JSONDecodeError:
        st.error("Error: Invalid JSON format in symptoms.json.")
        return []

# Match symptoms using NLP
def match_symptoms(user_input, symptoms_data):
    doc = nlp(user_input.lower())
    matched_symptoms = []
    
    for symptom_entry in symptoms_data:
        symptom = symptom_entry["symptom"]
        synonyms = symptom_entry.get("synonyms", [])
        symptom_doc = nlp(symptom.lower())
        synonym_docs = [nlp(syn.lower()) for s in synonyms]
        
        # Check similarity with main symptom and synonyms
        if any(doc.similarity(symptom_doc) > 0.8 for token in doc):
            matched_symptoms.append(symptom_entry)
        else:
            for syn_doc in synonym_docs:
                if any(doc.similarity(syn_doc) > 0.8 for token in doc):
                    matched_symptoms.append(symptom_entry)
                    break
    
    return matched_symptoms

# Generate differential diagnosis
def generate_differential_diagnosis(matched_symptoms):
    disease_probs = defaultdict(float)
    for symptom in matched_symptoms:
        for disease in symptom["diseases"]:
            disease_probs[disease["name"]] += disease["probability"]
    
    # Normalize probabilities
    total = sum(disease_probs.values())
    if total > 0:
        for disease in disease_probs:
            disease_probs[disease] /= total
    
    # Sort diseases by probability
    sorted_diseases = sorted(disease_probs.items(), key=lambda x: x[1], reverse=True)
    return sorted_diseases

# Determine triage level
def determine_triage(matched_symptoms):
    severity_levels = {"high": 3, "moderate": 2, "low": 1}
    max_severity = 0
    for symptom in matched_symptoms:
        severity = severity_levels.get(symptom["severity"], 1)
        max_severity = max(max_severity, severity)
    
    if max_severity >= 3:
        return "Emergent: Seek immediate medical attention."
    elif max_severity == 2:
        return "Urgent: Consult a healthcare provider soon."
    else:
        return "Non-urgent: Monitor symptoms and consult a doctor if they persist."

# Streamlit app
def main():
    st.title("Differential Diagnosis Tool")
    st.write("Enter your symptoms below (e.g., 'I have a fever and chest pain') to get a differential diagnosis and triage recommendation.")
    
    # Load symptoms data
    symptoms_data = load_symptoms()
    if not symptoms_data:
        return
    
    # User input
    user_input = st.text_area("Your Symptoms", placeholder="Describe your symptoms here...")
    
    if st.button("Diagnose"):
        if not user_input:
            st.warning("Please enter at least one symptom.")
            return
        
        # Match symptoms
        matched_symptoms = match_symptoms(user_input, symptoms_data)
        
        if not matched_symptoms:
            st.error("No matching symptoms found. Please try again.")
            return
        
        # Generate differential diagnosis
        differential_diagnosis = generate_differential_diagnosis(matched_symptoms)
        st.subheader("Differential Diagnosis")
        for disease, prob in differential_diagnosis:
            st.write(f"- {disease}: {prob:.2%} likelihood")
        
        # Determine triage level
        triage = determine_triage(matched_symptoms)
        st.subheader("Triage Recommendation")
        st.write(triage)

if __name__ == "__main__":
    main()