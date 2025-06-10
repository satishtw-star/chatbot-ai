import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import json
import os
import time
import urllib.parse
import re
from playwright.sync_api import sync_playwright
import openai

class VAScraper:
    def __init__(self):
        self.base_url = "https://www.va.gov"
        self.visited_urls = set()
        self.excluded_patterns = [
            'javascript:',  # Skip javascript links
            'tel:',        # Skip telephone links
            'mailto:',     # Skip email links
            '.pdf',        # Skip PDF files
            '.doc',        # Skip doc files
            '#'           # Skip anchor links
        ]
        # Initialize Playwright
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.context = self.browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )

    def __del__(self):
        """Clean up Playwright resources when the scraper is destroyed"""
        if hasattr(self, 'context'):
            self.context.close()
        if hasattr(self, 'browser'):
            self.browser.close()
        if hasattr(self, 'playwright'):
            self.playwright.stop()

    def is_valid_url(self, url: str) -> bool:
        """
        Check if URL should be scraped
        """
        # Must be a VA.gov URL
        if not url.startswith(self.base_url):
            return False
            
        # Check against excluded patterns
        return not any(pattern in url.lower() for pattern in self.excluded_patterns)

    def clean_url(self, url: str) -> str:
        """
        Clean URL by removing fragments and query parameters
        """
        parsed = urllib.parse.urlparse(url)
        return urllib.parse.urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            '',
            '',
            ''
        ))

    def get_page_content(self, url: str) -> Dict:
        """
        Scrape content from a VA.gov page using Playwright for JavaScript rendering
        """
        try:
            # Add delay to be respectful to VA.gov servers
            time.sleep(1)
            
            print(f"[DEBUG] Loading page with Playwright: {url}")
            page = self.context.new_page()
            page.goto(url, wait_until='networkidle')
            
            # Wait for the main content to load
            page.wait_for_selector('main', timeout=10000)
            
            # Get the page content after JavaScript has executed
            content = page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove navigation elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Get main content
            main_content = soup.find('main')
            if not main_content:
                main_content = soup.find('article')
            if not main_content:
                main_content = soup.find('body')
            
            # Get title and content
            title = page.title()
            content = main_content.get_text(separator=' ', strip=True) if main_content else ""
            
            # Skip if content is too short
            if len(content) < 200:  # Minimum content length threshold
                print(f"[DEBUG] Skipping page with insufficient content: {url}")
                page.close()
                return None
            
            # Find all links on the page within the main content area
            page_links = []
            seen_urls = set()
            
            # Use Playwright to find links within the main content area
            # We'll target 'a' tags that are descendants of 'main' or 'article' elements
            # This ensures we get links that are part of the main content, even if dynamically loaded
            links = page.query_selector_all('main a[href], article a[href]')
            
            if not links:
                # Fallback to body if main/article not found, but prioritize main content
                links = page.query_selector_all('body a[href]')

            for link_element in links:
                try:
                    href = link_element.get_attribute('href')
                    if not href:
                        continue
                        
                    link_text = link_element.text_content().strip()
                    
                    # Skip if link text is too short
                    if len(link_text) < 5:
                        continue
                    
                    # Convert relative URLs to absolute
                    if href.startswith('/'):
                        full_url = self.base_url + href
                    elif href.startswith(self.base_url):
                        full_url = href
                    else:
                        continue
                    
                    # Clean the URL
                    full_url = self.clean_url(full_url)
                    
                    # Only include valid VA.gov links that we haven't seen before
                    if self.is_valid_url(full_url) and full_url not in seen_urls:
                        seen_urls.add(full_url)
                        page_links.append({
                            "url": full_url,
                            "text": link_text
                        })
                except Exception as e:
                    print(f"[DEBUG] Error processing element: {str(e)}")
                    continue
            
            print(f"[DEBUG] Found {len(page_links)} links on page")
            page.close()
            
            return {
                "url": url,
                "title": title,
                "content": content,
                "links": page_links
            }
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    def crawl_va_pages(self, start_url: str, max_pages: int = 100) -> List[Dict]:
        """
        Crawl VA.gov starting from a specific URL. It will scrape the start_url
        and then all direct links found on that start_url, up to max_pages.
        """
        pages_data = []
        to_visit = []
        
        # Process the initial start_url first
        initial_url = self.clean_url(start_url)
        if initial_url not in self.visited_urls:
            self.visited_urls.add(initial_url)
            print(f"Scraping (0/{max_pages}): {initial_url}")
            page_data = self.get_page_content(initial_url)
            
            if page_data:
                pages_data.append(page_data)
                print(f"Successfully scraped page: {initial_url}")
                
                # Add links from the initial page to the to_visit list
                for link in page_data['links']:
                    url = link['url']
                    if self.is_valid_url(url) and url not in self.visited_urls and url not in to_visit:
                        to_visit.append(url)

        # Now process the collected links from the initial page
        while to_visit and len(pages_data) < max_pages:
            current_url = to_visit.pop(0)
            current_url = self.clean_url(current_url)
            
            if current_url in self.visited_urls:
                continue
                
            self.visited_urls.add(current_url)
            
            print(f"Scraping ({len(pages_data)}/{max_pages}): {current_url}")
            page_data = self.get_page_content(current_url)
            
            if page_data:
                pages_data.append(page_data)
                print(f"Successfully scraped page: {current_url}")

        return pages_data

    def save_to_json(self, data: List[Dict], filename: str):
        """
        Save scraped data to a JSON file
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    scraper = VAScraper()
    
    # Load start_urls from config file
    with open("scraper_config.json", "r") as f:
        config = json.load(f)
    start_urls = config.get("start_urls", ["https://www.va.gov/sign-in/"])
    
    all_pages_data = []
    for url in start_urls:
        print(f"\nStarting crawl from: {url}")
        pages_data = scraper.crawl_va_pages(url, max_pages=100)
        all_pages_data.extend(pages_data)
        print(f"Completed crawl from {url}. Total pages so far: {len(all_pages_data)}")
    
    print(f"\nScraping completed. Total pages scraped: {len(all_pages_data)}")
    scraper.save_to_json(all_pages_data, "va_content.json")

    # Example usage of the new OpenAI API
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="your text here"
    )
    embedding = response.data[0].embedding 