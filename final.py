import requests
import time
import json
import speech_recognition as sr
from pydub import AudioSegment
import re
import pandas as pd
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import pinecone
from huggingface_hub import login
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import spacy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from word2number import w2n
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import pickle
from datetime import datetime
from langdetect import detect
from googletrans import Translator
from spacy.language import Language

# Base URL for audio files
BASE_URL =   # Replace with your actual base URL
OUTPUT_JSON = "voice.json"
IMPORTED_JSON = "imp.json"

# Pinecone configuration
PINECONE_API_KEY = 
PINECONE_INDEX_NAME = 

# Add HuggingFace token configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings
HUGGINGFACE_TOKEN =   # Replace with your token

# Login to Hugging Face (you'll need to set your token)
login(token=HUGGINGFACE_TOKEN)

# Configure tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings

# Change the model to a simpler, publicly available one
model = SentenceTransformer('intfloat/multilingual-e5-large')

# Initialize language tools
translator = Translator()

# Dictionary of supported languages and their spaCy models
SUPPORTED_LANGUAGES = {
    'en': 'en_core_web_lg',
    'hi': 'xx_ent_wiki_sm',  # Hindi
    'mr': 'xx_ent_wiki_sm',  # Marathi
    'gu': 'xx_ent_wiki_sm',  # Gujarati
    # Add more languages as needed
}

# Load spaCy models for supported languages
nlp_models = {}
for lang_code, model_name in SUPPORTED_LANGUAGES.items():
    try:
        if lang_code == 'en':
            nlp_models[lang_code] = spacy.load(model_name)
        else:
            # For non-English languages, use multilingual model
            nlp_models[lang_code] = spacy.load('xx_ent_wiki_sm')
    except OSError:
        # Download if model is not installed
        spacy.cli.download(model_name)
        if lang_code == 'en':
            nlp_models[lang_code] = spacy.load(model_name)
        else:
            nlp_models[lang_code] = spacy.load('xx_ent_wiki_sm')

class SmartExtractor:
    def __init__(self):
        self.history_file = "extraction_history.json"
        self.model_file = "extractor_model.pkl"
        self.vectorizer_file = "vectorizer.pkl"
        self.history = self.load_history()
        self.model, self.vectorizer = self.load_or_create_model()
        
        # Add this: Initialize with some dummy data if no history exists
        if len(self.history['examples']) < 2:
            self.initialize_vectorizer()

    def initialize_vectorizer(self):
        """Initialize vectorizer with some dummy data to avoid NotFittedError"""
        dummy_texts = [
            "ignition parking switch off product number 36AF0018",
            "product number 12AB3456 quantity 10",
            "ID 1234 product name switch",
            "ignition switch product number 45CD6789"
        ]
        self.vectorizer.fit(dummy_texts)
        
    def load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {'examples': []}

    def load_or_create_model(self):
        if os.path.exists(self.model_file) and os.path.exists(self.vectorizer_file):
            with open(self.model_file, 'rb') as f:
                model = pickle.load(f)
            with open(self.vectorizer_file, 'rb') as f:
                vectorizer = pickle.load(f)
        else:
            model = RandomForestClassifier()
            vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        return model, vectorizer

    def save_model(self):
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_file, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def update_history(self, text, extracted_info, is_correct=True):
        self.history['examples'].append({
            'text': text,
            'extracted_info': extracted_info,
            'is_correct': is_correct,
            'timestamp': datetime.now().isoformat()
        })
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def train_model(self):
        if len(self.history['examples']) < 2:
            return

        texts = [ex['text'] for ex in self.history['examples']]
        X = self.vectorizer.fit_transform(texts)
        
        # Train separate models for each field
        self.field_models = {}
        for field in ['product_number', 'id', 'product_name', 'quantity']:
            y = [ex['extracted_info'].get(field) for ex in self.history['examples']]
            if len(set(y)) > 1:  # Only train if we have different values
                self.field_models[field] = RandomForestClassifier()
                self.field_models[field].fit(X, y)

    def extract_important_info(self, transcription):
        print(f"Processing text: {transcription}")
        
        # First try pattern-based extraction
        extracted_info = self.pattern_based_extraction(transcription)
        
        # Only use ML-based extraction if pattern matching didn't find everything
        if not all(extracted_info.values()):
            try:
                if len(self.history['examples']) >= 2:
                    X = self.vectorizer.transform([transcription])
                    for field, model in getattr(self, 'field_models', {}).items():
                        if not extracted_info.get(field) and field in self.field_models:
                            try:
                                prediction = model.predict(X)[0]
                                if prediction:
                                    extracted_info[field] = prediction
                            except Exception as e:
                                print(f"Warning: ML prediction failed for {field}: {e}")
            except Exception as e:
                print(f"Warning: ML-based extraction failed: {e}")

        # Post-process and validate
        extracted_info = self.post_process(extracted_info)
        
        # Update history with new example
        self.update_history(transcription, extracted_info)
        
        # Retrain model with new data
        try:
            self.train_model()
            self.save_model()
        except Exception as e:
            print(f"Warning: Model training failed: {e}")

        return extracted_info

    def pattern_based_extraction(self, text):
        extracted = {
            'product_number': None,
            'id': None,
            'product_name': None,
            'quantity': None
        }
        
        # First try to extract quantity and numbers (as they're more structured)
        quantity_patterns = [
            r'(?i)quantity\s*[:=]?\s*(\d+)',
            r'(?i)qty\s*[:=]?\s*(\d+)',
            r'(?i)(\d+)\s+(?:pieces?|pcs|units?)'
        ]
        
        for pattern in quantity_patterns:
            match = re.search(pattern, text)
            if match:
                extracted['quantity'] = match.group(1)
                # Remove the matched quantity to avoid confusion
                text = text.replace(match.group(0), '')
                break
        
        # Product number patterns
        product_number_patterns = [
            r'(?i)(\d{2})\s*([a-zA-Z])\s*([a-zA-Z])\s*(\d{4})',
            r'(?i)(\d{2}[a-zA-Z]{2}\d{4})'
        ]
        
        for pattern in product_number_patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) > 1:
                    parts = match.groups()
                    extracted['product_number'] = ''.join(parts).upper()
                else:
                    extracted['product_number'] = match.group(1)
                text = text.replace(match.group(0), '')
                break
        
        # ID patterns
        id_patterns = [
            r'(?i)(?:id|number)\s*[:=#]?\s*(\d{4})',
            r'(?i)\b(\d{4})\b'
        ]
        
        for pattern in id_patterns:
            match = re.search(pattern, text)
            if match:
                extracted['id'] = match.group(1)
                text = text.replace(match.group(0), '')
                break
        
        # Product name extraction - more flexible approach
        # Remove common filler words
        text = re.sub(r'(?i)\b(hey|hello|hi|need|want|a|an|the|please|with|of|for)\b', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if text:  # If there's any text left, it's likely the product name
            extracted['product_name'] = text.strip()
        
        return extracted

    def post_process(self, extracted_info):
        # Clean and validate extracted information
        if extracted_info.get('product_number'):
            extracted_info['product_number'] = re.sub(r'\s+', '', extracted_info['product_number']).upper()
        
        if extracted_info.get('id'):
            extracted_info['id'] = re.sub(r'\D', '', extracted_info['id'])
            
        if extracted_info.get('quantity'):
            extracted_info['quantity'] = re.sub(r'\D', '', extracted_info['quantity'])
        
        return extracted_info

def fetch_audio_files():
    response = requests.get(BASE_URL)
    if response.status_code == 200:
        audio_data = response.json()
        audio_files = [item['audio_file_url'] for item in audio_data if 'audio_file_url' in item]
        print("Fetched audio files:", audio_files)
        return audio_files
    else:
        print("Failed to fetch audio files.")
        return []

def convert_webm_to_wav(webm_url):
    response = requests.get(webm_url)
    webm_file_path = "temp_audio.webm"
    with open(webm_file_path, "wb") as webm_file:
        webm_file.write(response.content)

    wav_file_path = "temp_audio.wav"
    audio = AudioSegment.from_file(webm_file_path, format="webm")
    audio.export(wav_file_path, format="wav")
    return wav_file_path

def convert_audio_to_text(audio_url):
    recognizer = sr.Recognizer()
    try:
        wav_file_path = convert_webm_to_wav(audio_url)
        with sr.AudioFile(wav_file_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        print(f"Error processing {audio_url}: {e}")
        return ""

def update_json_file(text):
    with open(OUTPUT_JSON, "w") as json_file:
        json.dump({"transcription": text}, json_file)

def translate_to_english(text):
    """Translate text to English for processing"""
    try:
        # Detect language
        lang = detect(text)
        print(f"Detected language: {lang}")
        
        if lang == 'en':
            return text, lang
        
        # Use googletrans for translation
        translated_text = translator.translate(text, dest='en').text
        
        print(f"Original text: {text}")
        print(f"Translated text: {translated_text}")
        
        return translated_text, lang
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text, 'en'

def translate_back_to_original(text, original_lang):
    """Translate text back to original language"""
    try:
        if original_lang == 'en':
            return text
        
        translated_text = translator.translate(text, dest=original_lang).text
        return translated_text
    except Exception as e:
        print(f"Translation back error: {str(e)}")
        return text

def extract_multiple_products(transcription):
    print(f"Processing multilingual text: {transcription}")
    
    # Translate to English for processing
    english_text, original_lang = translate_to_english(transcription)
    
    # Split patterns for multiple products (in multiple languages)
    separators = {
        'en': ['and', 'also', 'along with', 'plus'],
        'hi': ['और', 'भी', 'के साथ', 'तथा'],
        'mr': ['आणि', 'सुद्धा', 'सोबत', 'बरोबर'],
        'gu': ['અને', 'પણ', 'સાથે', 'તેમજ']
    }
    
    # Get separators for detected language
    lang_separators = separators.get(original_lang, separators['en'])
    
    # Split text into segments
    product_segments = []
    current_text = english_text
    
    # Try splitting by language-specific separators
    for separator in lang_separators:
        if separator in current_text.lower():
            segments = [seg.strip() for seg in current_text.split(separator) if seg.strip()]
            if segments:
                product_segments.extend(segments)
                break
    
    # If no segments found, treat as single product
    if not product_segments:
        product_segments = [english_text]
    
    # Process each segment
    products_info = []
    for segment in product_segments:
        if segment.strip():
            extracted_info = extract_important_info(segment)
            if any(extracted_info.values()):
                # Translate product name back to original language if found
                if extracted_info.get('product_name'):
                    extracted_info['product_name'] = translate_back_to_original(
                        extracted_info['product_name'], 
                        original_lang
                    )
                products_info.append(extracted_info)
    
    print(f"Extracted {len(products_info)} products: {products_info}")
    return products_info

def extract_important_info(text):
    """Enhanced extraction with multilingual support"""
    # Translate to English for processing if needed
    english_text, original_lang = translate_to_english(text)
    
    # Use appropriate spaCy model
    nlp = nlp_models.get(original_lang, nlp_models['en'])
    doc = nlp(english_text)
    
    extracted_info = {
        'product_number': None,
        'id': None,
        'product_name': None,
        'quantity': None
    }
    
    # Extract using existing patterns (now working with English text)
    patterns = {
        'product_number': [
            r'(?i)(\d{2}[A-Za-z]{2}\d{4})',
            r'(?i)(\d{2})\s*([A-Za-z])\s*([A-Za-z])\s*(\d{4})'
        ],
        'id': [
            r'(?i)id\s*[:=#]?\s*(\d{4})',
            r'(?i)\b(\d{4})\b'
        ],
        'quantity': [
            r'(?i)quantity\s*[:=]?\s*(\d+)',
            r'(?i)qty\s*[:=]?\s*(\d+)',
            r'(\d+)\s*(?:pieces?|pcs|units?)'
        ]
    }
    
    # Extract using patterns
    for field, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, english_text)
            if match:
                if field == 'product_number' and len(match.groups()) > 1:
                    extracted_info[field] = ''.join(match.groups()).upper()
                else:
                    extracted_info[field] = match.group(1)
                break
    
    # Extract product name using NER
    if not extracted_info['product_name']:
        product_entities = [ent.text for ent in doc.ents if ent.label_ in ['PRODUCT', 'ORG']]
        if product_entities:
            extracted_info['product_name'] = product_entities[0]
    
    # Translate product name back to original language if needed
    if extracted_info['product_name'] and original_lang != 'en':
        extracted_info['product_name'] = translate_back_to_original(
            extracted_info['product_name'],
            original_lang
        )
    
    return extracted_info

def update_imported_json(products_info):
    print(f"Updating {IMPORTED_JSON} with products: {products_info}")
    with open(IMPORTED_JSON, "w") as json_file:
        json.dump({"products": products_info}, json_file, indent=2)
    print(f"Updated {IMPORTED_JSON}")

def connect_to_pinecone():
    try:
        # Create a Pinecone instance
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Get the index directly
        index = pc.Index(PINECONE_INDEX_NAME)
        print("Successfully connected to Pinecone index")
        return index
    
    except Exception as e:
        print(f"Error connecting to Pinecone: {str(e)}")
        raise

def vectorize_product_number(product_number):
    # Generate a vector for the product number
    vector = model.encode(product_number).tolist()  # Convert to list for compatibility
    return vector

def process_batch(batch_df, model):
    # Process a batch of rows at once
    texts = [' '.join(str(val) for val in row.values if pd.notna(val)) 
            for _, row in batch_df.iterrows()]
    
    # Encode all texts in the batch at once
    vectors = model.encode(texts, show_progress_bar=False)
    
    return [{
        'id': str(idx + batch_df.index[0]),
        'values': vector.tolist(),
        'metadata': {str(k): str(v) for k, v in row.to_dict().items()}
    } for idx, (vector, (_, row)) in enumerate(zip(vectors, batch_df.iterrows()))]

def create_final_json(result_json_path="result.json", final_json_path="final.json", imp_json_path="imp.json"):
    print("Creating final JSON with important information...")
    
    try:
        # Read the result.json file
        with open(result_json_path, 'r', encoding='utf-8') as file:
            results = json.load(file)
            print("Loaded results:", results)  # Debug print
            
        # Read the imp.json file for quantity
        with open(imp_json_path, 'r', encoding='utf-8') as file:
            imp_data = json.load(file)
            extracted_quantity = imp_data.get('quantity')
            print("Extracted quantity:", extracted_quantity)  # Debug print

        # Initialize empty final results
        final_results = {
            "product_number": "",
            "name": "",
            "id": "",
            "mrp": "",
            "quantity": str(extracted_quantity or '')
        }

        # Check if we have valid results
        if results and isinstance(results, list) and len(results) > 0:
            first_result = results[0]
            
            # Check if we have matches
            matches = first_result.get('results', {}).get('matches', [])
            if matches and len(matches) > 0:
                # Get the first (best) match
                best_match = matches[0]
                
                # Get the data dictionary
                data = best_match.get('data', {})
                print("Best match data:", data)  # Debug print
                
                # Update final results with found data
                final_results.update({
                    "product_number": str(data.get('Product_No', '')),
                    "name": str(data.get('Description', '')),
                    "id": str(data.get('Id', '')),
                    "mrp": str(data.get('Mrp', '')),
                    "quantity": str(extracted_quantity or '')
                })

        print("Final results:", final_results)  # Debug print

        # Save to final.json with nice formatting
        with open(final_json_path, 'w', encoding='utf-8') as file:
            json.dump(final_results, file, indent=2, ensure_ascii=False)
        
        # Post final.json to the API
        api_url = "https://bizonet.azurewebsites.net/api/erp_bizonet_demo/audio-order"
        try:
            response = requests.post(api_url, json=final_results)
            if response.status_code == 200:
                print(f"Successfully posted {final_json_path} to API")
            else:
                print(f"Failed to post {final_json_path} to API. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error posting {final_json_path} to API: {str(e)}")
        
        print(f"Successfully created {final_json_path}")
        return final_results

    except Exception as e:
        print(f"Error creating final JSON: {str(e)}")
        print("Full error:", e.__class__.__name__, str(e))  # More detailed error
        error_result = {
            "product_number": "",
            "name": "",
            "id": "",
            "mrp": "",
            "quantity": str(extracted_quantity if 'extracted_quantity' in locals() else '')
        }
        # Save error result to final.json
        with open(final_json_path, 'w', encoding='utf-8') as file:
            json.dump(error_result, file, indent=2, ensure_ascii=False)
        return error_result

def process_imp_json_and_search(excel_file):
    print("Starting search process for multiple products...")
    
    try:
        with open(IMPORTED_JSON, "r") as json_file:
            imported_data = json.load(json_file)
            products_list = imported_data.get("products", [])
        
        all_results = []
        for product_info in products_list:
            print(f"Processing product: {product_info}")
            
            excel_data = pd.read_excel(excel_file)
            search_text = None
            
            # Search priority remains the same for each product
            search_priority = ['product_number', 'product_name', 'id']
            
            for key in search_priority:
                if product_info.get(key):
                    search_text = str(product_info[key])
                    print(f"Using {key} as search term: {search_text}")
                    break
            
            if search_text:
                # Split search terms and remove common words
                search_terms = []
                if product_info.get('product_number'):
                    search_terms = [search_text]  # Don't split product numbers
                else:
                    # Remove common words and split into terms
                    cleaned_text = re.sub(r'(?i)\b(a|an|the|with|of|for)\b', ' ', search_text)
                    search_terms = [term.strip().lower() for term in cleaned_text.split() 
                                  if len(term.strip()) > 2]
                
                matches = []
                print(f"Searching for terms: {search_terms}")
                
                # Enhanced matching algorithm
                for idx, row in excel_data.iterrows():
                    row_text = ' '.join(str(val).lower() for val in row.values if pd.notna(val))
                    
                    # Different matching strategies based on search type
                    if product_info.get('product_number'):
                        # Exact matching for product numbers
                        if search_text.lower() in row_text:
                            matches.append({
                                'score': 1.0,
                                'metadata': row.to_dict(),
                                'match_type': 'exact'
                            })
                    else:
                        # Fuzzy matching for product names
                        match_score = 0
                        for term in search_terms:
                            # Check for exact word match
                            if f" {term} " in f" {row_text} ":
                                match_score += 1.0
                            # Check for partial word match
                            elif term in row_text:
                                match_score += 0.5
                            # Check for similar words using fuzzy matching
                            else:
                                for word in row_text.split():
                                    if fuzz.ratio(term, word) > 80:
                                        match_score += 0.3
                                        break
                        
                        if match_score > 0:
                            normalized_score = match_score / len(search_terms)
                            if normalized_score > 0.2:  # Lower threshold for better recall
                                matches.append({
                                    'score': normalized_score,
                                    'metadata': row.to_dict(),
                                    'match_type': 'exact' if normalized_score > 0.8 else 'partial'
                                })

                # Sort matches by score
                matches.sort(key=lambda x: x['score'], reverse=True)
                filtered_matches = matches[:10]  # Get top 10 matches

                if filtered_matches:
                    result = {
                        "search_term": search_text,
                        "total_matches": len(filtered_matches),
                        "confidence": "High" if filtered_matches[0]['score'] > 0.8 else "Medium",
                        "results": {
                            "matches": [
                                {
                                    "score": match['score'],
                                    "match_type": match['match_type'],
                                    "data": {
                                        "Product_No": str(match['metadata'].get('Product_No', '')),
                                        "Description": str(match['metadata'].get('Description', '')),
                                        "Id": str(match['metadata'].get('Id', '')),
                                        "Mrp": str(match['metadata'].get('Mrp', '')),
                                        "Quantity": str(product_info.get('quantity', ''))
                                    }
                                } for match in filtered_matches
                            ]
                        }
                    }
                    all_results.append(result)
                else:
                    print(f"No matches found for search term: {search_text}")
                    all_results.append({
                        "search_term": search_text,
                        "total_matches": 0,
                        "confidence": "Low",
                        "results": {"matches": []}
                    })
        
        # Save all results to result.json
        with open("result.json", "w", encoding='utf-8') as result_file:
            json.dump(all_results, result_file, indent=2, ensure_ascii=False)
        
        print("Results saved to result.json")
        
        # Create final.json with multiple products
        create_final_json_multiple(all_results)
        
        return all_results

    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def create_final_json_multiple(results, final_json_path="final.json"):
    try:
        final_products = []
        
        for result in results:
            if result.get('results', {}).get('matches'):
                best_match = result['results']['matches'][0]
                product_data = best_match['data']
                
                final_products.append({
                    "product_number": str(product_data.get('Product_No', '')),
                    "name": str(product_data.get('Description', '')),
                    "id": str(product_data.get('Id', '')),
                    "mrp": str(product_data.get('Mrp', '')),
                    "quantity": str(product_data.get('Quantity', ''))
                })
        
        final_results = {
            "order_details": {
                "total_products": len(final_products),
                "products": final_products
            }
        }
        
        # Save to final.json
        with open(final_json_path, 'w', encoding='utf-8') as file:
            json.dump(final_results, file, indent=2, ensure_ascii=False)
        
        # Post to API
        api_url = 
        try:
            response = requests.post(api_url, json=final_results)
            if response.status_code == 200:
                print(f"Successfully posted {final_json_path} to API")
            else:
                print(f"Failed to post {final_json_path} to API. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error posting {final_json_path} to API: {str(e)}")
        
        return final_results
    
    except Exception as e:
        print(f"Error creating final JSON: {str(e)}")
        return None

def main():
    try:
        # Fetch audio files
        audio_files = fetch_audio_files()
        
        if not audio_files:
            print("No audio files found.")
            return
        
        # Process each audio file
        for audio_url in audio_files:
            # Convert audio to text
            text = convert_audio_to_text(audio_url)
            if text:
                print(f"Transcribed text: {text}")
                
                # Update voice.json with transcription
                update_json_file(text)
                
                # Extract products information using SmartExtractor
                extractor = SmartExtractor()
                products_info = extract_multiple_products(text)
                
                # Update imp.json with extracted information
                update_imported_json(products_info)
                
                # Connect to Pinecone
                index = connect_to_pinecone()
                
                # Process the extracted information and search in Excel
                if os.path.exists("products.xlsx"):
                    results = process_imp_json_and_search("products.xlsx")
                    print("Search results:", results)
                    
                    # Create final JSON with the results
                    final_results = create_final_json()
                    print("Final results:", final_results)
                else:
                    print("Products database file not found.")
            
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        print("Full error details:", e.__class__.__name__, str(e))

if __name__ == "__main__":
    main()
