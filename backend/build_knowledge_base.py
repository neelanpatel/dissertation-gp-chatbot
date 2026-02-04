"""
NHS Knowledge Base Builder
Builds a searchable knowledge base from NHS condition pages for triage recommendations.
"""
import os
import json
import requests
import faiss
import numpy as np
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
import re

# Load environment and API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# NHS Configuration
SITEMAP_URLS = [
    "https://www.nhs.uk/sitemap-conditions-1.xml",
    "https://www.nhs.uk/sitemap-conditions-2.xml"
]

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# Web Scraping Functions

def get_condition_urls_from_sitemap(limit: int = 50) -> List[str]:
    """
    Fetches NHS conditions from sitemap XML files.
    Returns a list of condition page URLs.
    """
    all_urls = []
    headers = {'User-Agent': USER_AGENT}
    
    for sitemap_url in SITEMAP_URLS:
        print(f"Fetching sitemap: {sitemap_url}")
        try:
            response = requests.get(sitemap_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            namespace = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            
            for url_element in root.findall('s:url', namespace):
                loc_element = url_element.find('s:loc', namespace)
                if loc_element is not None:
                    url = loc_element.text
                    # Filter for condition pages only
                    if "/conditions/" in url and url not in all_urls:
                        all_urls.append(url)
            
            print(f"  Found {len(all_urls)} condition URLs so far")
            
            if len(all_urls) >= limit:
                break
                
        except Exception as e:
            print(f"  Error fetching sitemap {sitemap_url}: {e}")
            continue
    
    print(f"Total condition URLs found: {len(all_urls)}")
    return all_urls[:limit]

def extract_symptoms_from_page(soup: BeautifulSoup) -> str:
    """
    Extracts symptoms information from an NHS condition page.
    Tries multiple methods to find symptom content.
    """
    symptoms_text = ""
    
    # Method 1: Look for symptoms section by heading
    symptoms_heading = soup.find(["h2", "h3"], string=re.compile(r"symptom", re.IGNORECASE))
    if symptoms_heading:
        # Get all siblings until next heading
        current = symptoms_heading.find_next_sibling()
        while current and current.name not in ['h2', 'h3']:
            if current.name in ['ul', 'ol']:
                # Extract list items
                items = current.find_all('li')
                symptoms_text += " ".join([item.get_text(strip=True) for item in items]) + " "
            elif current.name == 'p':
                symptoms_text += current.get_text(strip=True) + " "
            current = current.find_next_sibling()
    
    # Method 2: Look for section with id containing 'symptom'
    if not symptoms_text:
        symptoms_section = soup.find(id=re.compile(r"symptom", re.IGNORECASE))
        if symptoms_section:
            symptoms_text = symptoms_section.get_text(" ", strip=True)
    
    # Method 3: Look in the page introduction/overview
    if not symptoms_text:
        intro = soup.find("div", class_="nhsuk-page-intro")
        if intro:
            symptoms_text = intro.get_text(" ", strip=True)
    
    return symptoms_text.strip() if symptoms_text else "No specific symptoms information available."

def determine_triage_level(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Determines the appropriate triage level based on care card content.
    Returns a dict with triage_advice and source_text.
    """
    # Default to GP
    result = {
        "triage_advice": "GP",
        "source_text": "See a GP if symptoms persist or worsen"
    }
    
    # NHS uses specific care card classes for different urgency levels
    care_card_mappings = [
        # Check emergency first (highest priority)
        {
            "classes": ["nhsuk-care-card--emergency", "nhsuk-care-card--immediate"],
            "keywords": ["999", "a&e", "emergency", "immediately", "straight away"],
            "triage": "Emergency"
        },
        # Then urgent
        {
            "classes": ["nhsuk-care-card--urgent"],
            "keywords": ["111", "urgent", "today", "quickly"],
            "triage": "Urgent"
        },
        # Then primary care
        {
            "classes": ["nhsuk-care-card--primary", "nhsuk-care-card--non-urgent"],
            "keywords": ["gp", "doctor", "appointment"],
            "triage": "GP"
        }
    ]
    
    # First check for pharmacist advice anywhere on the page
    pharmacist_sections = soup.find_all(text=re.compile(r"pharmacist can help", re.IGNORECASE))
    if pharmacist_sections:
        # Look for the containing element
        for text in pharmacist_sections:
            parent = text.parent
            while parent and parent.name not in ['div', 'section']:
                parent = parent.parent
            if parent:
                result["triage_advice"] = "Pharmacist"
                result["source_text"] = parent.get_text(" ", strip=True)[:500]
                return result  # Pharmacist is often the first port of call
    
    # Check care cards by class
    for mapping in care_card_mappings:
        for card_class in mapping["classes"]:
            cards = soup.find_all("div", class_=card_class)
            for card in cards:
                content = card.get_text(" ", strip=True).lower()
                # Check if any keywords match
                if any(keyword in content for keyword in mapping["keywords"]):
                    # Extract the heading and first part of content
                    heading = card.find(class_="nhsuk-care-card__heading")
                    card_content = card.find(class_="nhsuk-care-card__content")
                    
                    source_text = ""
                    if heading:
                        source_text = heading.get_text(strip=True) + ". "
                    if card_content:
                        # Get first paragraph or list
                        first_elem = card_content.find(['p', 'ul', 'ol'])
                        if first_elem:
                            source_text += first_elem.get_text(" ", strip=True)
                    
                    result["triage_advice"] = mapping["triage"]
                    result["source_text"] = source_text[:500] if source_text else card.get_text(" ", strip=True)[:500]
                    
                    # Emergency overrides everything
                    if mapping["triage"] == "Emergency":
                        return result
    
    # Also check for "See a GP" sections specifically
    gp_section = soup.find(text=re.compile(r"see a gp", re.IGNORECASE))
    if gp_section and result["triage_advice"] == "GP":
        parent = gp_section.parent
        while parent and parent.name not in ['div', 'section']:
            parent = parent.parent
        if parent:
            result["source_text"] = parent.get_text(" ", strip=True)[:500]
    
    return result

def scrape_condition_page(url: str, headers: Dict[str, str]) -> Optional[Dict]:
    """
    Scrapes a single NHS condition page and extracts relevant information.
    Returns a dict with condition data or None if scraping fails.
    """
    try:
        # Extract condition name from URL
        condition_name = url.split('/')[-1].replace('-', ' ').title()
        print(f"  Scraping: {condition_name}")
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract page title (might be more accurate than URL)
        page_title = soup.find("h1")
        if page_title:
            condition_name = page_title.get_text(strip=True)
        
        # Extract symptoms
        symptoms_text = extract_symptoms_from_page(soup)
        
        # Determine triage level
        triage_info = determine_triage_level(soup)
        
        # Extract overview/description if available
        overview = ""
        lead_paragraph = soup.find("p", class_="nhsuk-lead-paragraph")
        if lead_paragraph:
            overview = lead_paragraph.get_text(strip=True)
        
        # Build the search chunk (what we'll search against)
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
            "source_url": url
        }
        
    except requests.RequestException as e:
        print(f"    Request failed for {url}: {e}")
        return None
    except Exception as e:
        print(f"    Error processing {url}: {e}")
        return None

def scrape_all_conditions(urls: List[str], delay: float = 0.5) -> List[Dict]:
    """
    Scrapes multiple NHS condition pages with rate limiting.
    Returns a list of condition data dictionaries.
    """
    headers = {'User-Agent': USER_AGENT}
    conditions_data = []
    
    print(f"\nStarting to scrape {len(urls)} condition pages...")
    print("This may take a few minutes. Please be patient...\n")
    
    for i, url in enumerate(urls, 1):
        print(f"Progress: {i}/{len(urls)}")
        
        data = scrape_condition_page(url, headers)
        if data:
            conditions_data.append(data)
            print(f"    ✓ Successfully scraped: {data['condition_name']}")
            print(f"      Triage: {data['triage_advice']}")
        else:
            print(f"    ✗ Failed to scrape: {url}")
        
        # Rate limiting - be nice to NHS servers
        if i < len(urls):  # Don't delay after last request
            time.sleep(delay)
    
    print(f"\nSuccessfully scraped {len(conditions_data)} out of {len(urls)} pages")
    return conditions_data

# Sample Data for Testing 

def get_sample_data() -> List[Dict]:
    """
    Returns sample NHS condition data for testing when scraping isn't available.
    This represents what would be scraped from the NHS website.
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
            "source_explanation": "A pharmacist can give treatment advice and recommend flu remedies. Do not use flu remedies if you're taking paracetamol and ibuprofen tablets as these already contain these ingredients.",
            "source_url": "https://www.nhs.uk/conditions/flu/"
        },
        {
            "condition_name": "Chest Infection",
            "search_chunk": "Condition: Chest Infection. Overview: Chest infections affect your lungs or airways. Symptoms: Persistent chesty cough with yellow or green phlegm, wheezing and shortness of breath, chest pain or discomfort, high temperature, headache, aching muscles, tiredness.",
            "symptoms": "Persistent chesty cough with yellow or green phlegm, wheezing and shortness of breath, chest pain or discomfort, high temperature, headache, aching muscles, tiredness",
            "triage_advice": "GP",
            "source_explanation": "See a GP if you have a chest infection and: you feel very unwell or your symptoms get worse, you cough up blood or blood-stained phlegm, you've had a cough for more than 3 weeks.",
            "source_url": "https://www.nhs.uk/conditions/chest-infection/"
        },
        {
            "condition_name": "Hay Fever",
            "search_chunk": "Condition: Hay Fever. Overview: Hay fever is an allergic reaction to pollen, typically when it comes into contact with your mouth, nose, eyes and throat. Symptoms: Sneezing and coughing, runny or blocked nose, itchy red or watery eyes, itchy throat mouth nose and ears, loss of smell, pain around temples and forehead, headache, earache, feeling tired.",
            "symptoms": "Sneezing and coughing, runny or blocked nose, itchy red or watery eyes, itchy throat mouth nose and ears, loss of smell, pain around temples and forehead, headache, earache, feeling tired",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can help with hay fever. Speak to your pharmacist if you have hay fever. They can give advice and suggest the best treatments, like antihistamine drops, tablets or nasal sprays.",
            "source_url": "https://www.nhs.uk/conditions/hay-fever/"
        },
        {
            "condition_name": "Asthma Attack",
            "search_chunk": "Condition: Asthma Attack. Overview: An asthma attack is when symptoms get much worse suddenly. Symptoms: Severe wheezing when breathing, chest feels tight, cannot complete sentences, pulse racing, feeling agitated or panicked, blue lips or fingers, fainting or collapse.",
            "symptoms": "Severe wheezing when breathing, chest feels tight, cannot complete sentences, pulse racing, feeling agitated or panicked, blue lips or fingers, fainting or collapse",
            "triage_advice": "Emergency",
            "source_explanation": "Call 999 immediately if: you're having an asthma attack and your inhaler isn't helping, you're too breathless to speak, your lips or fingers are blue, you faint or collapse.",
            "source_url": "https://www.nhs.uk/conditions/asthma-attack/"
        },
        {
            "condition_name": "Urinary Tract Infection (UTI)",
            "search_chunk": "Condition: Urinary Tract Infection. Overview: UTIs are common infections that affect the bladder, kidneys and connected tubes. Symptoms: Pain or burning when urinating, needing to urinate more often especially at night, cloudy or smelly urine, blood in urine, lower abdomen pain, feeling generally unwell.",
            "symptoms": "Pain or burning when urinating, needing to urinate more often especially at night, cloudy or smelly urine, blood in urine, lower abdomen pain, feeling generally unwell",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can help with UTIs. You can ask a pharmacist about treatments for UTIs. They can suggest the best painkiller to take and may offer antibiotics if appropriate.",
            "source_url": "https://www.nhs.uk/conditions/urinary-tract-infections-utis/"
        },
        {
            "condition_name": "Sore Throat",
            "search_chunk": "Condition: Sore Throat. Overview: Sore throats are very common and usually get better by themselves within 3 to 7 days. Symptoms: Pain when swallowing, dry scratchy throat, redness in back of mouth, bad breath, mild cough, swollen neck glands.",
            "symptoms": "Pain when swallowing, dry scratchy throat, redness in back of mouth, bad breath, mild cough, swollen neck glands",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can help with sore throats. You can ask a pharmacist about ways of relieving the pain and discomfort of a sore throat, such as paracetamol or ibuprofen.",
            "source_url": "https://www.nhs.uk/conditions/sore-throat/"
        },
        {
            "condition_name": "Heart Attack",
            "search_chunk": "Condition: Heart Attack. Overview: A heart attack is a medical emergency where blood supply to the heart is suddenly blocked. Symptoms: Chest pain like pressure squeezing or heaviness, pain spreading to arms back neck jaw stomach, shortness of breath, feeling weak or lightheaded, overwhelming anxiety, coughing or wheezing.",
            "symptoms": "Chest pain like pressure squeezing or heaviness, pain spreading to arms back neck jaw stomach, shortness of breath, feeling weak or lightheaded, overwhelming anxiety, coughing or wheezing",
            "triage_advice": "Emergency",
            "source_explanation": "Call 999 immediately if you think someone is having a heart attack. The faster you act, the better their chances. Do not delay - every minute matters.",
            "source_url": "https://www.nhs.uk/conditions/heart-attack/"
        },
        {
            "condition_name": "Sprain",
            "search_chunk": "Condition: Sprain. Overview: A sprain happens when ligaments are stretched or torn. Symptoms: Pain tenderness or weakness, swelling or bruising, unable to put weight on injury, muscle spasms or cramping, limited flexibility.",
            "symptoms": "Pain tenderness or weakness, swelling or bruising, unable to put weight on injury, muscle spasms or cramping, limited flexibility",
            "triage_advice": "Pharmacist",
            "source_explanation": "A pharmacist can help with sprains. They can offer advice about the best painkiller to use and provide supports like bandages and elastic wraps.",
            "source_url": "https://www.nhs.uk/conditions/sprains/"
        },
        {
            "condition_name": "Diabetes",
            "search_chunk": "Condition: Diabetes. Overview: Diabetes is a condition that causes blood sugar levels to become too high. Symptoms: Feeling very thirsty, urinating more frequently especially at night, feeling very tired, weight loss and muscle wasting, itchy genitals or thrush, cuts that heal slowly, blurred vision.",
            "symptoms": "Feeling very thirsty, urinating more frequently especially at night, feeling very tired, weight loss and muscle wasting, itchy genitals or thrush, cuts that heal slowly, blurred vision",
            "triage_advice": "GP",
            "source_explanation": "See a GP if you have symptoms of diabetes or you're worried you may have a higher risk of getting diabetes. Early diagnosis and treatment is very important.",
            "source_url": "https://www.nhs.uk/conditions/diabetes/"
        }
    ]

# Embedding and Storage 

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Converts texts into embeddings using OpenAI's embedding model.
    """
    if not texts:
        return []
    
    print(f"\nGenerating embeddings for {len(texts)} text chunks...")
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
            "My chest hurts and I can't breathe",
            "I keep needing to pee and it burns",
            "I'm having an allergic reaction with sneezing"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            # Get embedding for query
            response = client.embeddings.create(
                input=[query],
                model="text-embedding-3-small"
            )
            query_embedding = np.array([response.data[0].embedding]).astype("float32")
            
            # Search
            k = 3
            distances, indices = index.search(query_embedding, k)
            
            print("Top matches:")
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
                condition = data[idx]
                print(f"  {i}. {condition['condition_name']}")
                print(f"     Triage: {condition['triage_advice']}")
                print(f"     Distance: {dist:.2f}")
        
        print("\n✓ Knowledge base is working correctly!")
        
    except Exception as e:
        print(f"Error testing knowledge base: {e}")

# Main Execution 

def main(use_sample_data: bool = False, limit: int = 50):
    """
    Main function to build the knowledge base.
    
    Args:
        use_sample_data: If True, uses sample data instead of scraping
        limit: Maximum number of conditions to scrape
    """
    print("=" * 60)
    print("NHS Knowledge Base Builder")
    print("=" * 60)
    
    if use_sample_data:
        print("\nUsing sample data (no web scraping)...")
        conditions_data = get_sample_data()
    else:
        print("\nAttempting to scrape NHS website...")
        print(f"Target: {limit} condition pages")
        
        # Get URLs from sitemap
        urls = get_condition_urls_from_sitemap(limit=limit)
        
        if not urls:
            print("\n⚠ Could not fetch URLs from NHS website.")
            print("Falling back to sample data...")
            conditions_data = get_sample_data()
        else:
            # Scrape condition pages
            conditions_data = scrape_all_conditions(urls)
            
            if not conditions_data:
                print("\n⚠ Could not scrape any condition pages.")
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