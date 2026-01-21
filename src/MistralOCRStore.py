# (COMPLETED PASS)

# MistralOCRStore.py
# This is used to extract the full-text of scientific paper PDFs using the Mistral OCR API, with caching.


import os
import json
import time
import base64

from mistralai import Mistral

# Thread lock
from threading import Lock
mistral_cache_thread_lock = Lock()

MISTRAL_OCR_CACHE_FOLDER = "ocr-cache/"

class MistralOCRStore:
    def __init__(self, api_key:str=None):
        # Check if the API key is set
        if (api_key == None):
            # Check if the environment variable is set
            api_key = os.environ.get("MISTRAL_API_KEY", None)
            if (api_key == None):
                # Try to load the API key from a file
                api_key = self.load_api_key("api_keys.donotcommit.json")

        if (api_key == None):
            raise ValueError("Mistral API key is not set.")

        print ("First two characters of Mistral API key: " + api_key[:2] + "**")

        # Initialize the Mistral client
        self.client = Mistral(api_key=api_key)

        # Check that the cache folder exists, if not create it
        if (not os.path.exists(MISTRAL_OCR_CACHE_FOLDER)):
            os.makedirs(MISTRAL_OCR_CACHE_FOLDER)


    # Load the API key from a JSON file
    def load_api_key(self, file_path:str):
        # Load the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
        if ("mistral" in data):
            return data["mistral"]

        return None


    def sanitize_url(self, url:str):
        # Strip off any leading "http://" or "https://"
        if (url.startswith("http://")):
            url = url[7:]
        elif (url.startswith("https://")):
            url = url[8:]

        # Convert anything that isn't a letter, number, or underscore to an underscore
        sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in url)
        # Remove leading and trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized


    def load_cache(self, cached_url:str):
        # Acquire the thread lock
        with mistral_cache_thread_lock:
            # Sanitize the URL to create a valid filename
            sanitized_url = self.sanitize_url(cached_url)

            # Construct the directory for this cache url
            cache_file_path = os.path.join(MISTRAL_OCR_CACHE_FOLDER, sanitized_url)
            # Check if the cache directory exists
            if (not os.path.exists(cache_file_path)):
                return None

            # Check if the cache file exists
            cache_file_path = os.path.join(cache_file_path, "ocr_response.md")
            if (not os.path.exists(cache_file_path)):
                return None

            # Load the cached response
            try:
                # Load the Markdown file
                with open(cache_file_path, 'r') as file:
                    markdown_str = file.read()
                print(f"Cache loaded for {cached_url}")
                return markdown_str
            except Exception as e:
                print(f"Error loading cache for {cached_url}: {e}")
                return None


    def save_cache(self, cached_url:str, ocr_response_json:dict):
        # Acquire the thread lock
        with mistral_cache_thread_lock:

            # Sanitize the URL to create a valid filename
            sanitized_url = self.sanitize_url(cached_url)

            # Construct the directory for this cache url
            cache_file_path = os.path.join(MISTRAL_OCR_CACHE_FOLDER, sanitized_url)
            # Check if the cache directory exists, if not create it
            if (not os.path.exists(cache_file_path)):
                os.makedirs(cache_file_path)

            # Save the response to a file
            cache_file = os.path.join(cache_file_path, "ocr_response.json")
            try:
                with open(cache_file, 'w') as file:
                    json.dump(ocr_response_json, file)
                print(f"Cache saved for {cached_url}")
            except Exception as e:
                print(f"Error saving cache for {cached_url}: {e}")

            # Also try to save a markdown version of the response
            markdown_str = self.save_markdown_from_ocr_response(ocr_response_json, cache_file_path)
            return markdown_str



    def get_markdown_from_ocr_response(self, ocr_response:dict):
        # Extract the markdown content from the OCR response
        markdown_str = ""
        if ("pages" in ocr_response):
            for page in ocr_response.get("pages", []):
                try:
                    if ("markdown" in page):
                        markdown_str += page["markdown"] + "\n\n"
                except Exception as e:
                    print(f"Error processing page {page.get('id', 'unknown')}: {e}")

        return markdown_str.strip()


    def save_markdown_from_ocr_response(self, ocr_response:dict, file_path:str):
        # Save the markdown content to a file
        markdown_str = self.get_markdown_from_ocr_response(ocr_response)
        try:
            markdown_filename = os.path.join(file_path, "ocr_response.md")
            with open(markdown_filename, 'w') as file:
                file.write(markdown_str)
            print(f"Markdown saved to {markdown_filename}")
        except Exception as e:
            print(f"Error saving markdown to {markdown_filename}: {e}")

        # Also save the images
        for page in ocr_response.get("pages", []):
            for img in page.get("images", []):
                img_id = None
                try:
                    img_id = img['id']
                    data_uri = img['image_base64']
                    # Split out the base64-encoded data
                    _, encoded = data_uri.split(",", 1)
                    # Decode and write to file
                    img_data = base64.b64decode(encoded)
                    image_path = os.path.join(file_path, img_id)
                    with open(image_path, "wb") as f:
                        f.write(img_data)
                    print(f"Saved {image_path}")
                except Exception as e:
                    print(f"Error saving image {img['id']}: {e}")

        # Return the markdown string
        return markdown_str.strip()

    # Process a PDF
    def process_pdf(self, pdf_url:str, max_pages:int=20):
        print("process_pdf: started for url: " + str(pdf_url))

        try:
            # Step 1: First, check for a cached version
            cached_response = self.load_cache(pdf_url)
            if (cached_response is not None):
                print(f"Cache hit for {pdf_url}")
                return cached_response


            # Step 2: If no cached version, call the Mistral OCR API
            start_time = time.time()
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": pdf_url,
                },
                include_image_base64=True,
                pages=range(0, max_pages + 1),  # Process the first `max_pages` pages
            )
            processing_time = time.time() - start_time
            print(f"OCR processing time for {pdf_url}: {processing_time:.2f} seconds")

            # Step 3: Save the response to a cache
            # Convert the response to JSON
            ocr_response_json = ocr_response.dict()
            markdown_str = self.save_cache(pdf_url, ocr_response_json)

            # Return the markdown
            return markdown_str


        except Exception as e:
            print(f"Error processing PDF {pdf_url}: {e}")
            return None




# Example stand-alone usage
if __name__ == "__main__":
    # Initialize the MistralOCRStore with your API key
    ocr_store = MistralOCRStore(api_key=None)

    # Process a PDF URL
    pdf_url = "https://arxiv.org/pdf/2503.22708"
    response = ocr_store.process_pdf(pdf_url)

    # Print the response
    print(json.dumps(response, indent=4))