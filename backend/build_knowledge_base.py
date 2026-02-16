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

# Load environment and API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Add NHS API Key 
NHS_API_KEY = os.getenv("NHS_API_KEY") 

# NHS API Configuration
API_BASE_URL = "https://int.api.service.nhs.uk/nhs-website-content"


# API Fetching Functions

def get_condition_urls_from_api(limit: int = 50) -> List[str]:
    """
    Fetches the master list of NHS conditions from the API.
    Returns a list of specific condition API endpoints.
    """
    all_urls = []
    headers = {'apikey': NHS_API_KEY}
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
    """
    Extracts symptoms information from the structured JSON modules.
    """
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
    """
    Determines the appropriate triage level based on NHS clinical keywords.
    Returns a dict with triage_advice and source_text.
    """
    # Default to GP
    result = {
        "triage_advice": "GP",
        "source_text": "See a GP if symptoms persist or worsen"
    }
    
    # NHS uses specific keywords for different urgency levels
    care_card_mappings = [
        # Check emergency first (highest priority)
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
    """
    Fetches a single NHS condition via API and extracts relevant information.
    Returns a dict with condition data or None if fetching fails.
    """
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
    """
    Iterates through API endpoints to download condition data while respecting rate limits.
    """
    headers = {'apikey': NHS_API_KEY}
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
        
        # Rate limiting (0.6s ensures we stay safely under 120 per minute)
        if i < len(urls):  
            time.sleep(delay)
    
    print(f"\nSuccessfully downloaded {len(conditions_data)} out of {len(urls)} condition records")
    return conditions_data

# Sample Data for Fallback

def get_sample_data() -> List[Dict]:
    """
    Returns sample NHS condition data for testing if the API is unavailable.
    """
    return [
        {
            "condition_name": "Common Cold",
            "search_chunk": "Condition: Common Cold. Overview: The common cold is a viral infection of your nose and throat. Symptoms: Blocked or runny nose, sore throat, headaches, muscle aches, cough, sneezing, raised temperature, pressure in ears and face, loss of taste and smell.",
            "symptoms": "Blocked or runny nose, sore throat, headaches, muscle aches, cough, sneezing, raised temperature, pressure in ears and face, loss of taste and smell",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can help with cold medicines. You can buy cough and cold medicines from pharmacies or supermarkets. A pharmacist can advise you on the best medicine.",
            "source_url": "https://www.nhs.uk/conditions/common-cold/"
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
            "condition_name": "Chest Infection",
            "search_chunk": "Condition: Chest Infection. Overview: Chest infections affect your lungs or airways. Symptoms: Persistent chesty cough with yellow or green phlegm, wheezing and shortness of breath, chest pain or discomfort, high temperature, headache, aching muscles, tiredness.",
            "symptoms": "Persistent chesty cough with yellow or green phlegm, wheezing and shortness of breath, chest pain or discomfort, high temperature, headache, aching muscles, tiredness",
            "triage_advice": "GP",
            "source_explanation": "See a GP if you have a chest infection and you feel very unwell.",
            "source_url": "https://www.nhs.uk/conditions/chest-infection/"
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
    """
    Tests the knowledge base with sample queries.
    """
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
    """
    Main function to build the knowledge base.
    
    Args:
        use_sample_data: If True, uses sample data instead of fetching
        limit: Maximum number of conditions to fetch
    """
    print("=" * 60)
    print("NHS API Data Ingestion Pipeline")
    print("=" * 60)
    
    if not NHS_API_KEY and not use_sample_data:
        print("⚠ ERROR: NHS_API_KEY not found in environment variables.")
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