"""
NHS Knowledge Base Builder
Fetches structured medical data from the NHS Website Content API v2 
to build a searchable vector knowledge base for triage recommendations.
"""
import os
import json
import requests
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import time
from typing import List, Dict, Optional
import re


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

NHS_CLIENT_ID = os.getenv("NHS_CLIENT_ID")
NHS_CLIENT_SECRET = os.getenv("NHS_CLIENT_SECRET")
TOKEN_URL = "https://int.api.service.nhs.uk/oauth2/token"

# Add NHS API Key 
NHS_API_KEY = os.getenv("NHS_API_KEY") 

# NHS API Configuration
API_BASE_URL = "https://int.api.service.nhs.uk/nhs-website-content"


def get_access_token() -> Optional[str]:
    try:
        response = requests.post(
            TOKEN_URL,
            data={
                'grant_type': 'client_credentials',
                'client_id': NHS_CLIENT_ID,
                'client_secret': NHS_CLIENT_SECRET
            },
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=10
        )
        response.raise_for_status()
        return response.json().get('access_token')
    except Exception as e:
        print(f"  ✗ Failed to get access token: {e}")
        print(f"  Response: {e.response.text if hasattr(e, 'response') else 'No response body'}")
        return None


# API Fetching Functions

def get_condition_urls_from_api(limit: int = 50) -> List[str]:

    # Fetches the master list of NHS conditions from the API & Returns a list of specific condition API endpoints

    all_urls = []
    token = get_access_token()
    if not token:
        return []
    headers = {'Authorization': f'Bearer {token}'}
    list_endpoint = f"{API_BASE_URL}/conditions"
    
    print(f"Connecting to NHS API: {list_endpoint}")
    try:
        response = requests.get(list_endpoint, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # The API returns a list of conditions inside 'hasPart'
        items = data.get('hasPart', []) if isinstance(data, dict) else data
        
        for item in items:
            url = item.get('url', '')
            if url and url not in all_urls:
                all_urls.append(url)
                
            if len(all_urls) >= limit:
                break
                
        print(f"  ✓ Located {len(all_urls)} condition endpoints")
            
    except Exception as e:
        print(f"  ✗ Error fetching conditions list: {e}")
    
    return all_urls[:limit]

def extract_symptoms_from_api(modules: List[Dict]) -> str:

    # Extracts symptoms information from the structured JSON modules.

    symptoms_text = ""
    
    for module in modules:
        module_name = module.get('name', '').lower()
        module_text = module.get('text', '') or module.get('description', '')
        
        # Method 1: Look for the dedicated symptoms module
        if 'symptom' in module_name:
            # Strip any residual HTML tags from the API text
            clean_text = re.sub(r'<[^>]+>', ' ', module_text)
            symptoms_text += clean_text + " "
            
    # Method 2: Fallback to the page introduction/overview if no specific symptoms module exists
    if not symptoms_text and modules:
        overview_text = modules[0].get('text', '') or modules[0].get('description', '')
        clean_text = re.sub(r'<[^>]+>', ' ', overview_text)
        symptoms_text = clean_text
    
    return symptoms_text.strip() if symptoms_text else "No specific symptoms information available."

def determine_triage_level_from_api(modules: List[Dict], full_text: str) -> Dict[str, str]:

    # Determines the appropriate triage level based on NHS clinical keywords.


    # Default to GP
    result = {
        "triage_advice": "GP",
        "source_text": "See a GP if symptoms persist or worsen"
    }
    
    # NHS uses specific keywords for different urgency levels
    care_card_mappings = [
        # Check emergency first 
        {
            "keywords": ["call 999", "go to a&e", "emergency", "immediately", "straight away", "999"],
            "triage": "Emergency"
        },
        # Then urgent
        {
            "keywords": ["call 111", "urgent", "today", "quickly", "111"],
            "triage": "Urgent"
        },
        # Then primary care
        {
            "keywords": ["see a gp", "doctor", "appointment"],
            "triage": "GP"
        }
    ]
    
    lower_text = full_text.lower()
    
    # First check for pharmacist advice anywhere on the page
    if "pharmacist can help" in lower_text or "speak to a pharmacist" in lower_text:
        result["triage_advice"] = "Pharmacist"
        match = re.search(r'([^.]*pharmacist[^.]*\.)', full_text, re.IGNORECASE)
        result["source_text"] = match.group(1).strip() if match else "A pharmacist can provide advice and treatments."
        return result
    
    # Check keyword mappings
    for mapping in care_card_mappings:
        for keyword in mapping["keywords"]:
            if keyword in lower_text:
                result["triage_advice"] = mapping["triage"]
                # Grab the sentence containing the keyword for context
                match = re.search(r'([^.]*' + re.escape(keyword) + r'[^.]*\.)', full_text, re.IGNORECASE)
                if match:
                    result["source_text"] = match.group(1).strip()
                
                if mapping["triage"] == "Emergency":
                    return result
    
    return result

def fetch_condition_data(url: str, headers: Dict[str, str]) -> Optional[Dict]:

    # Fetches a single NHS condition via API and extracts relevant information.


    try:
        # Append modules=true for AI-friendly chunking
        fetch_url = url if 'modules=true' in url else f"{url}?modules=true"
        print(f"  Fetching: {url.split('/')[-1]}")
        
        response = requests.get(fetch_url, headers=headers, timeout=10)
        
        # Handle 120 req/min rate limit gracefully
        if response.status_code == 429:
            print("    Rate limit hit. Waiting 5 seconds...")
            time.sleep(5)
            response = requests.get(fetch_url, headers=headers, timeout=10)
            
        response.raise_for_status()
        data = response.json()
        
        condition_name = data.get('name', url.split('/')[-1].replace('-', ' ').title())
        modules = data.get('hasPart', [])
        
        # Combine all text to search for triage keywords
        full_text = condition_name + " "
        for m in modules:
            full_text += str(m.get('name', '')) + " " + str(m.get('description', '')) + " " + str(m.get('text', '')) + " "
        full_text = re.sub(r'<[^>]+>', ' ', full_text) # Strip HTML
        
        # Extract symptoms and triage
        symptoms_text = extract_symptoms_from_api(modules)
        triage_info = determine_triage_level_from_api(modules, full_text)
        
        # Extract overview/description
        overview = data.get('description', '')
        if not overview and modules:
            overview_raw = modules[0].get('description', '') or modules[0].get('text', '')
            overview = re.sub(r'<[^>]+>', ' ', overview_raw)[:200]
            
        # Build the search chunk
        search_chunk = f"Condition: {condition_name}. "
        if overview:
            search_chunk += f"Overview: {overview}. "
        search_chunk += f"Symptoms: {symptoms_text}"
        
        return {
            "condition_name": condition_name,
            "search_chunk": search_chunk,
            "symptoms": symptoms_text,
            "triage_advice": triage_info["triage_advice"],
            "source_explanation": triage_info["source_text"],
            "source_url": data.get('url', url)
        }
        
    except requests.RequestException as e:
        print(f"    Request failed for {url}: {e}")
        return None
    except Exception as e:
        print(f"    Error processing {url}: {e}")
        return None

def fetch_all_conditions(urls: List[str], delay: float = 0.6) -> List[Dict]:

    # Iterates through API endpoints to download condition data while respecting rate limits.

    token = get_access_token()
    if not token:
        return []
    headers = {'Authorization': f'Bearer {token}'}
    conditions_data = []
    
    print(f"\nStarting API ingestion for {len(urls)} conditions...")
    print("This may take a few minutes depending on rate limits...\n")
    
    for i, url in enumerate(urls, 1):
        print(f"Progress: {i}/{len(urls)}")
        
        data = fetch_condition_data(url, headers)
        if data:
            conditions_data.append(data)
            print(f"    ✓ Successfully processed: {data['condition_name']}")
        else:
            print(f"    ✗ Failed to process: {url}")
        
        # Rate limiting 
        if i < len(urls):  
            time.sleep(delay)
    
    print(f"\nSuccessfully downloaded {len(conditions_data)} out of {len(urls)} condition records")
    return conditions_data

# Sample Data for Fallback

def get_sample_data() -> List[Dict]:

    # Returns sample NHS condition data for testing if the API is unavailable.

    return [
        {
            "condition_name": "Chest Infection",
            "search_chunk": "Condition: Chest Infection. Overview: Chest infections affect your lungs or airways. Symptoms: Persistent chesty cough with yellow or green phlegm, wheezing, high temperature.",
            "symptoms": "Persistent chesty cough with yellow or green phlegm, wheezing, high temperature",
            "triage_advice": "GP",
            "source_explanation": "See a GP if you have a chest infection and you feel very unwell or your symptoms get worse.",
            "source_url": "https://www.nhs.uk/conditions/chest-infection/"
        },
        {
            "condition_name": "Back Pain",
            "search_chunk": "Condition: Back Pain. Overview: Back pain is very common and usually improves within a few weeks. Symptoms: Aching or stiffness in the back, sharp pain in the neck or lower back.",
            "symptoms": "Aching or stiffness in the back, sharp pain in the neck or lower back",
            "triage_advice": "GP",
            "source_explanation": "See a GP if your back pain does not improve after a few weeks or is severe.",
            "source_url": "https://www.nhs.uk/conditions/back-pain/"
        },
        {
            "condition_name": "Chest Infection",
            "search_chunk": "Condition: Chest Infection. Overview: Chest infections affect your lungs or airways. Symptoms: Persistent chesty cough with yellow or green phlegm, wheezing and shortness of breath, chest pain or discomfort, high temperature, headache, aching muscles, tiredness.",
            "symptoms": "Persistent chesty cough with yellow or green phlegm, wheezing and shortness of breath, chest pain or discomfort, high temperature, headache, aching muscles, tiredness",
            "triage_advice": "GP",
            "source_explanation": "See a GP if you have a chest infection and you feel very unwell.",
            "source_url": "https://www.nhs.uk/conditions/chest-infection/"
        },
        {
            "condition_name": "Shingles",
            "search_chunk": "Condition: Shingles. Overview: Shingles is an infection that causes a painful rash. Symptoms: Painful, red, blistery rash on your arm, body, or face, tingling or burning feeling, headache, feeling generally unwell.",
            "symptoms": "Painful, red, blistery rash on your arm, body, or face, tingling or burning feeling, headache, feeling generally unwell",
            "triage_advice": "Pharmacist (if 18 or over) or GP (if under 18)",
            "source_explanation": "If you are 18 or over, a pharmacist can help with shingles. If you are under 18, you should see a GP. You need treatment within 3 days of the rash appearing.",
            "source_url": "https://www.nhs.uk/conditions/shingles/"
        },
        {
            "condition_name": "Stroke",
            "search_chunk": "Condition: Stroke. Overview: A stroke is a serious life-threatening medical condition that happens when the blood supply to part of the brain is cut off. Symptoms: Face drooping on one side, unable to smile, cannot lift both arms and keep them there, slurred speech or difficulty understanding what someone is saying.",
            "symptoms": "Face drooping on one side, unable to smile, cannot lift both arms and keep them there, slurred speech or difficulty understanding what someone is saying",
            "triage_advice": "Emergency",
            "source_explanation": "Call 999 immediately if you notice any single one of the signs of a stroke (Face, Arms, Speech, Time).",
            "source_url": "https://www.nhs.uk/conditions/stroke/"
        },
        {
            "condition_name": "Meningitis",
            "search_chunk": "Condition: Meningitis. Overview: Meningitis is an infection of the protective membranes that surround the brain and spinal cord. Symptoms: High temperature, cold hands and feet, stiff neck, dislike of bright lights, a rash that does not fade when a glass is rolled over it.",
            "symptoms": "High temperature, cold hands and feet, stiff neck, dislike of bright lights, a rash that does not fade when a glass is rolled over it",
            "triage_advice": "Emergency",
            "source_explanation": "Call 999 or go to A&E immediately. Meningitis is a medical emergency. A rash that does not fade under a glass is a critical warning sign.",
            "source_url": "https://www.nhs.uk/conditions/meningitis/"
        },
        {
            "condition_name": "Sepsis",
            "search_chunk": "Condition: Sepsis. Overview: Sepsis is a life-threatening reaction to an infection. Symptoms: Blue, pale or blotchy skin, lips or tongue, a rash that does not fade, difficulty breathing, severe breathlessness, confusion, slurred speech.",
            "symptoms": "Blue, pale or blotchy skin, lips or tongue, a rash that does not fade, difficulty breathing, severe breathlessness, confusion, slurred speech",
            "triage_advice": "Emergency",
            "source_explanation": "Call 999 or go to A&E immediately if you have symptoms of sepsis. It is a life-threatening medical emergency.",
            "source_url": "https://www.nhs.uk/conditions/sepsis/"
        },
        {
            "condition_name": "Flu",
            "search_chunk": "Condition: Flu. Overview: Flu is a common infectious viral illness spread by coughs and sneezes. Symptoms: Sudden fever, body aches, feeling tired or exhausted, dry cough, sore throat, headache, difficulty sleeping, loss of appetite, stomach ache, feeling sick or being sick.",
            "symptoms": "Sudden fever, body aches, feeling tired or exhausted, dry cough, sore throat, headache, difficulty sleeping, loss of appetite, stomach ache, feeling sick or being sick",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can give treatment advice and recommend flu remedies.",
            "source_url": "https://www.nhs.uk/conditions/flu/"
        },
         {
            "condition_name": "Common Cold",
            "search_chunk": "Condition: Common Cold. Overview: The common cold is a viral infection of your nose and throat. Symptoms: Blocked or runny nose, sore throat, headaches, muscle aches, cough, sneezing, raised temperature, pressure in ears and face, loss of taste and smell.",
            "symptoms": "Blocked or runny nose, sore throat, headaches, muscle aches, cough, sneezing, raised temperature, pressure in ears and face, loss of taste and smell",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can help with cold medicines. You can buy cough and cold medicines from pharmacies or supermarkets. A pharmacist can advise you on the best medicine.",
            "source_url": "https://www.nhs.uk/conditions/common-cold/"
        },
            {
            "condition_name": "Food Poisoning",
            "search_chunk": "Condition: Food Poisoning. Overview: Food poisoning is an illness caused by eating contaminated food. Symptoms: Feeling sick (nausea), diarrhoea, being sick (vomiting), stomach cramps, high temperature.",
            "symptoms": "Feeling sick (nausea), diarrhoea, being sick (vomiting), stomach cramps, high temperature",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can help with food poisoning by recommending oral rehydration sachets to prevent dehydration.",
            "source_url": "https://www.nhs.uk/conditions/food-poisoning/"
        },
        {
            "condition_name": "Conjunctivitis",
            "search_chunk": "Condition: Conjunctivitis. Overview: Conjunctivitis is an eye condition caused by infection or allergies. Symptoms: Red, itchy, sticky, or watery eyes, crust on your eyelashes.",
            "symptoms": "Red, itchy, sticky, or watery eyes, crust on your eyelashes",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist or optician can help with conjunctivitis. They can offer advice and provide eye drops or antihistamines.",
            "source_url": "https://www.nhs.uk/conditions/conjunctivitis/"
        },
        {
            "condition_name": "Toothache",
            "search_chunk": "Condition: Toothache. Overview: Tooth pain can be caused by tooth decay, a cracked tooth, or an infection. Symptoms: Continuous sharp or throbbing pain in your tooth or mouth, swelling around the tooth, pain when chewing.",
            "symptoms": "Continuous sharp or throbbing pain in your tooth or mouth, swelling around the tooth, pain when chewing",
            "triage_advice": "Dentist",
            "source_explanation": "A GP cannot provide dental treatment. You must see a dentist if you have a toothache that lasts more than 2 days.",
            "source_url": "https://www.nhs.uk/conditions/toothache/"
        },
        {
            "condition_name": "Hay Fever",
            "search_chunk": "Condition: Hay Fever. Overview: Hay fever is an allergic reaction to pollen. Symptoms: Sneezing, runny or blocked nose, itchy red or watery eyes, itchy throat, mouth, nose and ears, loss of smell, headache.",
            "symptoms": "Sneezing, runny or blocked nose, itchy red or watery eyes, itchy throat, mouth, nose and ears, loss of smell, headache",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can give advice and suggest the best treatments, such as antihistamine drops, tablets or nasal sprays.",
            "source_url": "https://www.nhs.uk/conditions/hay-fever/"
        },
        {
            "condition_name": "Urinary Tract Infection",
            "search_chunk": "Condition: Urinary Tract Infection. Overview: A UTI is an infection in the urinary system. Symptoms: Pain or burning when urinating, needing to urinate more often than usual, dark, cloudy or strong-smelling urine, pain in the lower tummy, feeling tired and unwell.",
            "symptoms": "Pain or burning when urinating, needing to urinate more often than usual, dark cloudy or strong-smelling urine, pain in the lower tummy, feeling tired and unwell",
            "triage_advice": "GP",
            "source_explanation": "See a GP if you think you have a UTI. A GP can prescribe antibiotics to treat the infection.",
            "source_url": "https://www.nhs.uk/conditions/urinary-tract-infections-utis/"
        },
        {
            "condition_name": "Asthma",
            "search_chunk": "Condition: Asthma. Overview: Asthma is a common lung condition that causes occasional breathing difficulties. Symptoms: Wheezing, breathlessness, a tight chest, coughing.",
            "symptoms": "Wheezing, breathlessness, a tight chest, coughing",
            "triage_advice": "GP",
            "source_explanation": "See a GP if you think you or your child might have asthma, or if your asthma is getting worse. Call 999 if you have a severe asthma attack.",
            "source_url": "https://www.nhs.uk/conditions/asthma/"
        },
        {
            "condition_name": "High Blood Pressure",
            "search_chunk": "Condition: High Blood Pressure (Hypertension). Overview: High blood pressure rarely has noticeable symptoms, but if untreated, it increases your risk of serious problems such as heart attacks and strokes. Symptoms: Headaches, blurred vision, shortness of breath, nosebleeds (if very high).",
            "symptoms": "Headaches, blurred vision, shortness of breath, nosebleeds",
            "triage_advice": "GP",
            "source_explanation": "Get your blood pressure checked by a GP or pharmacist. You can also test it at home. See a GP if your readings are consistently high.",
            "source_url": "https://www.nhs.uk/conditions/high-blood-pressure-hypertension/"
        },
        {
            "condition_name": "Type 2 Diabetes",
            "search_chunk": "Condition: Type 2 Diabetes. Overview: Type 2 diabetes is a common condition that causes the level of sugar in the blood to become too high. Symptoms: Peeing more than usual, feeling thirsty all the time, feeling very tired, losing weight without trying, blurred vision.",
            "symptoms": "Peeing more than usual, feeling thirsty all the time, feeling very tired, losing weight without trying, blurred vision",
            "triage_advice": "GP",
            "source_explanation": "See a GP if you have any of the symptoms of type 2 diabetes. You'll need a blood test to check.",
            "source_url": "https://www.nhs.uk/conditions/type-2-diabetes/"
        },
        {
            "condition_name": "Atopic Eczema",
            "search_chunk": "Condition: Atopic Eczema. Overview: Atopic eczema is a condition that causes the skin to become itchy, dry and cracked. Symptoms: Itchy, dry, cracked, and sore skin, often in the creases of joints like knees and elbows.",
            "symptoms": "Itchy, dry, cracked, and sore skin",
            "triage_advice": "Pharmacist",
            "source_explanation": "Speak to a pharmacist for advice and treatments like emollients. See a GP if it does not improve or looks infected.",
            "source_url": "https://www.nhs.uk/conditions/atopic-eczema/"
        },
        {
            "condition_name": "Gout",
            "search_chunk": "Condition: Gout. Overview: Gout is a type of arthritis that causes sudden, severe joint pain. Symptoms: Sudden, severe pain in a joint (usually the big toe), hot, swollen, red skin over the affected joint.",
            "symptoms": "Sudden severe pain in a joint, hot swollen red skin over the affected joint",
            "triage_advice": "GP",
            "source_explanation": "See a GP if you have sudden severe pain in a joint. It's important to get gout treated to prevent damage to joints.",
            "source_url": "https://www.nhs.uk/conditions/gout/"
        },
        {
            "condition_name": "Migraine",
            "search_chunk": "Condition: Migraine. Overview: A migraine is usually a moderate or severe headache felt as a throbbing pain on one side of the head. Symptoms: Throbbing head pain, feeling sick, being sick, increased sensitivity to light and sound.",
            "symptoms": "Throbbing head pain, feeling sick, being sick, increased sensitivity to light and sound",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can recommend painkillers. See a GP if your migraines are severe, frequent, or getting worse.",
            "source_url": "https://www.nhs.uk/conditions/migraine/"
        },
        {
            "condition_name": "Osteoarthritis",
            "search_chunk": "Condition: Osteoarthritis. Overview: Osteoarthritis is a condition that causes joints to become painful and stiff. Symptoms: Joint pain and stiffness, swelling, tenderness, grating or crackling sound when moving the affected joints.",
            "symptoms": "Joint pain and stiffness, swelling, tenderness, grating or crackling sound when moving",
            "triage_advice": "GP",
            "source_explanation": "See a GP if you have persistent symptoms of osteoarthritis so they can confirm the diagnosis and prescribe treatment.",
            "source_url": "https://www.nhs.uk/conditions/osteoarthritis/"
        },
        {
            "condition_name": "Rheumatoid Arthritis",
            "search_chunk": "Condition: Rheumatoid Arthritis. Overview: Rheumatoid arthritis is a long-term condition that causes pain, swelling and stiffness in the joints. Symptoms: Throbbing and aching joint pain, joint swelling, joint stiffness especially in the morning.",
            "symptoms": "Throbbing and aching joint pain, joint swelling, joint stiffness",
            "triage_advice": "GP",
            "source_explanation": "See a GP if you think you have symptoms of rheumatoid arthritis, so they can try to identify the underlying cause.",
            "source_url": "https://www.nhs.uk/conditions/rheumatoid-arthritis/"
        },
        {
            "condition_name": "Appendicitis",
            "search_chunk": "Condition: Appendicitis. Overview: Appendicitis is a painful swelling of the appendix. Symptoms: Pain in the middle of your tummy that travels to your lower right-hand side, feeling sick, being sick, loss of appetite, high temperature.",
            "symptoms": "Pain travelling to lower right tummy, feeling sick, being sick, loss of appetite, high temperature",
            "triage_advice": "Emergency",
            "source_explanation": "Call 999 or go to A&E immediately if you have sudden, severe pain in your tummy. Appendicitis requires urgent medical attention.",
            "source_url": "https://www.nhs.uk/conditions/appendicitis/"
        },
        {
            "condition_name": "Heart Attack",
            "search_chunk": "Condition: Heart Attack. Overview: A heart attack happens when the supply of blood to the heart is suddenly blocked. Symptoms: Chest pain (pressure, heaviness, tightness), pain spreading to the arms, neck, jaw, or back, shortness of breath, feeling dizzy.",
            "symptoms": "Chest pain or tightness, pain spreading to arms or jaw, shortness of breath, feeling dizzy",
            "triage_advice": "Emergency",
            "source_explanation": "Call 999 immediately. A heart attack is a serious medical emergency.",
            "source_url": "https://www.nhs.uk/conditions/heart-attack/"
        },
        {
            "condition_name": "Chickenpox",
            "search_chunk": "Condition: Chickenpox. Overview: Chickenpox is common and mostly affects children, but you can get it at any age. Symptoms: Itchy, spotty rash that turns into fluid-filled blisters, high temperature, aches, loss of appetite.",
            "symptoms": "Itchy spotty rash, fluid-filled blisters, high temperature, aches, loss of appetite",
            "triage_advice": "Pharmacist",
            "source_explanation": "Speak to a pharmacist about cooling creams and antihistamines. Call 111 if the skin around the spots becomes red, hot or painful.",
            "source_url": "https://www.nhs.uk/conditions/chickenpox/"
        },
        {
            "condition_name": "Measles",
            "search_chunk": "Condition: Measles. Overview: Measles is an infection that spreads very easily and can cause serious problems in some people. Symptoms: High temperature, a runny or blocked nose, sneezing, a cough, red, sore, watery eyes, spots in the mouth, a rash starting on the face.",
            "symptoms": "High temperature, runny nose, cough, red eyes, rash starting on the face",
            "triage_advice": "GP",
            "source_explanation": "Call your GP surgery before you go in if you think you or your child has measles. It's highly infectious.",
            "source_url": "https://www.nhs.uk/conditions/measles/"
        },
        {
            "condition_name": "Mumps",
            "search_chunk": "Condition: Mumps. Overview: Mumps is a contagious viral infection that used to be common in children. Symptoms: Painful swellings at the side of the face under the ears (parotid glands), headaches, joint pain, high temperature.",
            "symptoms": "Painful swellings at the side of the face under the ears, headaches, joint pain, high temperature",
            "triage_advice": "GP",
            "source_explanation": "Call your GP if you suspect mumps. It's important to rule out more serious infections like glandular fever or tonsillitis.",
            "source_url": "https://www.nhs.uk/conditions/mumps/"
        },
        {
            "condition_name": "Whooping Cough",
            "search_chunk": "Condition: Whooping Cough. Overview: Whooping cough (pertussis) is a bacterial infection of the lungs and breathing tubes. Symptoms: Bouts of coughing that last for a few minutes, a 'whoop' sound between coughs, bringing up thick mucus.",
            "symptoms": "Bouts of coughing, 'whoop' sound between coughs, bringing up thick mucus",
            "triage_advice": "GP",
            "source_explanation": "See a GP or call 111 urgently if you or your child have symptoms of whooping cough.",
            "source_url": "https://www.nhs.uk/conditions/whooping-cough/"
        },
        {
            "condition_name": "Pneumonia",
            "search_chunk": "Condition: Pneumonia. Overview: Pneumonia is swelling (inflammation) of the tissue in one or both lungs, usually caused by a bacterial infection. Symptoms: A cough that may be dry or produce thick yellow, green, brown or blood-stained mucus, difficulty breathing, rapid heartbeat, high temperature.",
            "symptoms": "Cough producing thick or discoloured mucus, difficulty breathing, rapid heartbeat, high temperature",
            "triage_advice": "GP",
            "source_explanation": "See a GP or call 111 if you have symptoms of pneumonia. Call 999 if you are struggling to breathe or have blue lips/fingers.",
            "source_url": "https://www.nhs.uk/conditions/pneumonia/"
        },
        {
            "condition_name": "Tonsillitis",
            "search_chunk": "Condition: Tonsillitis. Overview: Tonsillitis is an infection of the tonsils at the back of your throat. Symptoms: A sore throat, difficulty swallowing, hoarse or no voice, high temperature, coughing, a headache, feeling sick.",
            "symptoms": "Sore throat, difficulty swallowing, high temperature, coughing, headache",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can help with tonsillitis. See a GP if symptoms are severe, don't go away after 4 days, or you have white pus-filled spots on your tonsils.",
            "source_url": "https://www.nhs.uk/conditions/tonsillitis/"
        },
        {
            "condition_name": "Ear Infection",
            "search_chunk": "Condition: Ear Infection. Overview: Ear infections are very common, particularly in children. Symptoms: Pain inside the ear, high temperature, being sick, a lack of energy, difficulty hearing, discharge running out of the ear.",
            "symptoms": "Pain inside the ear, high temperature, being sick, lack of energy, difficulty hearing",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can advise on painkillers. See a GP if the pain is severe, lasts more than 3 days, or there's fluid coming from the ear.",
            "source_url": "https://www.nhs.uk/conditions/ear-infections/"
        },
        {
            "condition_name": "Thrush",
            "search_chunk": "Condition: Thrush. Overview: Thrush is a common yeast infection that affects men and women. Symptoms: White patches in the mouth or on the tongue, itching or irritation around the genitals, white discharge.",
            "symptoms": "White patches in mouth, genital itching or irritation, white discharge",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can recommend antifungal treatments. See a GP if it's your first time having thrush or if treatments don't work.",
            "source_url": "https://www.nhs.uk/conditions/thrush-in-men-and-women/"
        },
        {
            "condition_name": "Scabies",
            "search_chunk": "Condition: Scabies. Overview: Scabies is an itchy skin rash caused by mites. Symptoms: Intense itching, especially at night, a raised rash or spots, often starting between the fingers.",
            "symptoms": "Intense itching especially at night, raised rash or spots",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can recommend a cream or lotion to treat scabies. Everyone in your household needs to be treated.",
            "source_url": "https://www.nhs.uk/conditions/scabies/"
        },
        {
            "condition_name": "Head Lice and Nits",
            "search_chunk": "Condition: Head Lice and Nits. Overview: Head lice and nits are very common in young children and their families. Symptoms: An itchy scalp, feeling like something is moving in the hair, spotting the lice or empty white eggshells (nits).",
            "symptoms": "Itchy scalp, feeling movement in hair, visible lice or nits",
            "triage_advice": "Pharmacist",
            "source_explanation": "You can treat head lice without seeing a GP. A pharmacist can advise on lotions, sprays, and wet combing techniques.",
            "source_url": "https://www.nhs.uk/conditions/head-lice-and-nits/"
        },
        {
            "condition_name": "Threadworms",
            "search_chunk": "Condition: Threadworms. Overview: Threadworms are tiny worms in your poo. They're common in children and spread easily. Symptoms: Extreme itching around the anus or vagina, particularly at night, seeing tiny white worms in poo.",
            "symptoms": "Extreme itching around the anus or vagina at night, tiny white worms in poo",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can provide medicine for threadworms. The whole household must be treated, even if they don't have symptoms.",
            "source_url": "https://www.nhs.uk/conditions/threadworms/"
        },
        {
            "condition_name": "Ringworm",
            "search_chunk": "Condition: Ringworm. Overview: Ringworm is a common fungal infection. It's not caused by worms. Symptoms: A ring-shaped, red, silvery or scaly rash. It can occur on the face, body, groin, or scalp.",
            "symptoms": "Ring-shaped, red, silvery or scaly rash, itching",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can recommend antifungal creams, gels or sprays. See a GP if it's on your scalp or treatments do not work.",
            "source_url": "https://www.nhs.uk/conditions/ringworm/"
        },
        {
            "condition_name": "Athlete's Foot",
            "search_chunk": "Condition: Athlete's Foot. Overview: Athlete's foot is a common fungal infection that affects the feet. Symptoms: Itchy white patches between your toes, red, sore and flaky skin on your feet, skin that may crack or bleed.",
            "symptoms": "Itchy white patches between toes, red, sore and flaky skin, cracked skin",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can help with athlete's foot by recommending antifungal treatments.",
            "source_url": "https://www.nhs.uk/conditions/athletes-foot/"
        },
        {
            "condition_name": "Cold Sore",
            "search_chunk": "Condition: Cold Sore. Overview: Cold sores are common and usually clear up on their own within 10 days. Symptoms: A tingling, itching or burning feeling around your mouth, followed by small, fluid-filled blisters.",
            "symptoms": "Tingling, itching or burning feeling around the mouth, small fluid-filled blisters",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can recommend antiviral creams or patches. See a GP if the cold sore is very large or hasn't healed after 10 days.",
            "source_url": "https://www.nhs.uk/conditions/cold-sores/"
        },
        {
            "condition_name": "Warts and Verrucas",
            "search_chunk": "Condition: Warts and Verrucas. Overview: Warts and verrucas are small lumps on the skin that most people have at some point. Symptoms: Small, rough, fleshy lumps on the skin, often on hands (warts) or feet (verrucas).",
            "symptoms": "Small, rough, fleshy lumps on hands or feet",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can advise on creams, plasters and sprays to treat warts and verrucas. They can take months to clear.",
            "source_url": "https://www.nhs.uk/conditions/warts-and-verrucas/"
        },
        {
            "condition_name": "Piles",
            "search_chunk": "Condition: Piles (Haemorrhoids). Overview: Piles are lumps inside and around your bottom (anus). Symptoms: Bright red blood after you poo, an itchy anus, feeling like you still need to poo after going, lumps around your anus.",
            "symptoms": "Bright red blood after pooing, itchy anus, lumps around your anus",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can recommend creams to ease pain and itching. See a GP if symptoms don't improve after 7 days.",
            "source_url": "https://www.nhs.uk/conditions/piles-haemorrhoids/"
        },
        {
            "condition_name": "Constipation",
            "search_chunk": "Condition: Constipation. Overview: Constipation is a common condition that affects people of all ages. Symptoms: Not pooing at least 3 times a week, poo that is unusually large or small and dry, lumpy or hard, straining or in pain when you poo.",
            "symptoms": "Not pooing often, dry, lumpy or hard poo, straining or pain when pooing",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can recommend laxatives. See a GP if it does not improve, or you are bloated and have stomach pain.",
            "source_url": "https://www.nhs.uk/conditions/constipation/"
        },
        {
            "condition_name": "Diarrhoea and Vomiting",
            "search_chunk": "Condition: Diarrhoea and Vomiting. Overview: Diarrhoea and vomiting are common in adults and children and usually pass in a few days. Symptoms: Frequent watery poo, being sick, stomach cramps, high temperature.",
            "symptoms": "Frequent watery poo, being sick, stomach cramps, high temperature",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can recommend oral rehydration salts. Speak to 111 or a GP if symptoms last longer than a few days or you show signs of severe dehydration.",
            "source_url": "https://www.nhs.uk/conditions/diarrhoea-and-vomiting/"
        },
        {
            "condition_name": "Irritable Bowel Syndrome",
            "search_chunk": "Condition: Irritable Bowel Syndrome (IBS). Overview: IBS is a common condition that affects the digestive system. Symptoms: Stomach pain or cramps, bloating, diarrhoea, constipation.",
            "symptoms": "Stomach pain or cramps, bloating, diarrhoea, constipation",
            "triage_advice": "GP",
            "source_explanation": "See a GP if you think you have IBS symptoms, so they can rule out other conditions like coeliac disease.",
            "source_url": "https://www.nhs.uk/conditions/irritable-bowel-syndrome-ibs/"
        },
        {
            "condition_name": "Heartburn and Acid Reflux",
            "search_chunk": "Condition: Heartburn and Acid Reflux. Overview: Heartburn is an uncomfortable burning feeling in your chest that can move up to your neck and throat. Symptoms: Burning feeling in the chest, a sour taste in the mouth, repeated burping, hoarse voice.",
            "symptoms": "Burning feeling in the chest, sour taste in mouth, repeated burping",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can offer antacids. See a GP if you have heartburn most days for 3 weeks or more.",
            "source_url": "https://www.nhs.uk/conditions/heartburn-and-acid-reflux/"
        },
        {
            "condition_name": "Gallstones",
            "search_chunk": "Condition: Gallstones. Overview: Gallstones are small stones, usually made of cholesterol, that form in the gallbladder. Symptoms: Sudden, severe abdominal pain (biliary colic), usually in the center or upper right of the tummy, feeling sick, sweating.",
            "symptoms": "Sudden severe pain in upper right tummy, feeling sick, sweating",
            "triage_advice": "GP",
            "source_explanation": "See a GP if you experience gallstone symptoms. Go to A&E if the pain is so intense you cannot find a comfortable position, or if you turn yellow (jaundice).",
            "source_url": "https://www.nhs.uk/conditions/gallstones/"
        },
        {
            "condition_name": "Kidney Stones",
            "search_chunk": "Condition: Kidney Stones. Overview: Kidney stones can be extremely painful, and can lead to kidney infections or the kidney not working properly. Symptoms: Severe pain in the side of your tummy (abdomen) or groin, pain that comes and goes, feeling sick, blood in urine.",
            "symptoms": "Severe pain in the side of your tummy or groin, feeling sick, blood in urine",
            "triage_advice": "Emergency",
            "source_explanation": "Call 111 or go to A&E if you have severe pain, a high temperature, or an episode of shivering and shaking.",
            "source_url": "https://www.nhs.uk/conditions/kidney-stones/"
        },
        {
            "condition_name": "Deep Vein Thrombosis",
            "search_chunk": "Condition: Deep Vein Thrombosis (DVT). Overview: DVT is a blood clot in a vein, usually in the leg. Symptoms: Throbbing or cramping pain in 1 leg (rarely both), swelling in 1 leg, warm skin around the painful area, red or darkened skin around the painful area.",
            "symptoms": "Throbbing or cramping pain in 1 leg, swelling in 1 leg, warm or red skin around the painful area",
            "triage_advice": "Emergency",
            "source_explanation": "Call 111 or go to A&E immediately. DVT is a medical emergency that can lead to a pulmonary embolism.",
            "source_url": "https://www.nhs.uk/conditions/deep-vein-thrombosis-dvt/"
        },
        {
            "condition_name": "Iron Deficiency Anaemia",
            "search_chunk": "Condition: Iron Deficiency Anaemia. Overview: Iron deficiency anaemia is caused by a lack of iron, often because of blood loss or pregnancy. Symptoms: Tiredness and lack of energy, shortness of breath, noticeable heartbeats (heart palpitations), pale skin.",
            "symptoms": "Tiredness, lack of energy, shortness of breath, heart palpitations, pale skin",
            "triage_advice": "GP",
            "source_explanation": "See a GP. A simple blood test will confirm if you're anaemic and they can prescribe iron supplements.",
            "source_url": "https://www.nhs.uk/conditions/iron-deficiency-anaemia/"
        },
        {
            "condition_name": "Glandular Fever",
            "search_chunk": "Condition: Glandular Fever. Overview: Glandular fever is a viral infection that mostly affects teenagers and young adults. Symptoms: Severely sore throat, very high temperature, swollen glands in the neck, extreme tiredness.",
            "symptoms": "Severely sore throat, high temperature, swollen glands in the neck, extreme tiredness",
            "triage_advice": "GP",
            "source_explanation": "See a GP if you suspect glandular fever. They can arrange a blood test to confirm the diagnosis.",
            "source_url": "https://www.nhs.uk/conditions/glandular-fever/"
        },
        {
            "condition_name": "COVID-19",
            "search_chunk": "Condition: COVID-19. Overview: COVID-19 is a respiratory illness caused by a virus. Symptoms: High temperature, a new, continuous cough, a loss or change to your sense of smell or taste, shortness of breath, feeling tired.",
            "symptoms": "High temperature, new continuous cough, loss of sense of smell or taste, shortness of breath, feeling tired",
            "triage_advice": "111",
            "source_explanation": "Try to stay at home and avoid contact with others. Call 111 if your symptoms worsen or you are worried.",
            "source_url": "https://www.nhs.uk/conditions/covid-19/"
        }
        

    ]

# Embedding and Storage 

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Converts texts into embeddings using OpenAI's embedding model.
    """
    if not texts:
        return []
    
    print(f"\nGenerating OpenAI embeddings for {len(texts)} text chunks...")
    try:
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

def build_and_save_index(data: List[Dict]) -> bool:
    """
    Creates FAISS index and saves it along with the metadata.
    Returns True if successful.
    """
    if not data:
        print("No data to index.")
        return False
    
    # Extract text to embed
    search_texts = [item['search_chunk'] for item in data]
    
    # Generate embeddings
    embeddings = get_embeddings(search_texts)
    if not embeddings:
        print("Failed to generate embeddings.")
        return False
    
    print(f"Building FAISS index with {len(embeddings)} embeddings...")
    
    # Create FAISS index
    dimension = len(embeddings[0])
    embeddings_np = np.array(embeddings).astype("float32")
    
    # Using L2 distance for similarity
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    
    # Save index
    faiss.write_index(index, "nhs_index.faiss")
    print("✓ Saved FAISS index to 'nhs_index.faiss'")
    
    # Save metadata
    with open("nhs_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved metadata for {len(data)} conditions to 'nhs_data.json'")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Knowledge Base Summary:")
    print(f"  Total conditions: {len(data)}")
    
    triage_counts = {}
    for item in data:
        triage = item['triage_advice']
        triage_counts[triage] = triage_counts.get(triage, 0) + 1
    
    print("  Triage distribution:")
    for triage, count in sorted(triage_counts.items()):
        print(f"    - {triage}: {count} conditions")
    print("=" * 60)
    
    return True

# Testing Function

def test_knowledge_base():

    # Tests the knowledge base with sample queries.

    print("\n" + "=" * 60)
    print("Testing Knowledge Base")
    print("=" * 60)
    
    try:
        # Load the index and data
        index = faiss.read_index("nhs_index.faiss")
        with open("nhs_data.json", "r") as f:
            data = json.load(f)
        
        test_queries = [
            "I have a runny nose and sore throat",
            "My chest hurts and I can't breathe"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            # Get embedding for query
            response = client.embeddings.create(input=[query], model="text-embedding-3-small")
            query_embedding = np.array([response.data[0].embedding]).astype("float32")
            
            # Search
            distances, indices = index.search(query_embedding, 3)
            print("Top matches:")
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
                condition = data[idx]
                print(f"  {i}. {condition['condition_name']} (Triage: {condition['triage_advice']})")
        
        print("\n✓ Knowledge base is working correctly!")
    except Exception as e:
        print(f"Error testing knowledge base: {e}")

# Main Execution 

def main(use_sample_data: bool = False, limit: int = 50):
    
  
    print("=" * 60)
    print("NHS API Data Ingestion Pipeline")
    print("=" * 60)
    
    if not NHS_CLIENT_ID or not NHS_CLIENT_SECRET:
        print("⚠ ERROR: NHS_CLIENT_ID or NHS_CLIENT_SECRET not found in environment variables.")
        print("Falling back to sample data...")
        use_sample_data = True
    
    if use_sample_data:
        print("\nUsing fallback sample data...")
        conditions_data = get_sample_data()
    else:
        print("\nInitiating connection to NHS Website Content API...")
        print(f"Target: Fetching {limit} condition records")
        
        # Get URLs from API
        urls = get_condition_urls_from_api(limit=limit)
        
        if not urls:
            print("\n⚠ Could not retrieve API endpoints.")
            print("Falling back to sample data...")
            conditions_data = get_sample_data()
        else:
            conditions_data = fetch_all_conditions(urls)
            if not conditions_data:
                print("\n⚠ Failed to download condition data.")
                print("Falling back to sample data...")
                conditions_data = get_sample_data()
    
    # Build and save the index
    if build_and_save_index(conditions_data):
        print("\n✅ Knowledge base built successfully!")
        
        # Test the knowledge base
        if os.getenv("OPENAI_API_KEY"):
            test_knowledge_base()

if __name__ == "__main__":
    main()