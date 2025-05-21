import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import json
import os
import time
import urllib.parse
import re
from playwright.sync_api import sync_playwright

class VAScraper:
    def __init__(self):
        self.base_url = "https://www.va.gov"
        self.visited_urls = set()
        self.excluded_patterns = [
            '/directory/',  # Skip directory pages
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
        Scrape content from a VA.gov page, with special handling for the sign-in page using Playwright.
        """
        try:
            # Add delay to be respectful to VA.gov servers
            time.sleep(1)
            
            # Special handling for sign-in page using Playwright
            if '/sign-in/' in url:
                print(f"[DEBUG] Loading sign-in page with Playwright: {url}")
                page = self.context.new_page()
                page.goto(url, wait_until='networkidle')
                
                # Wait for the main content to load
                page.wait_for_selector('main', timeout=10000)
                
                # Get the page content after JavaScript has executed
                content = page.content()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Find all links on the page
                page_links = []
                seen_urls = set()
                
                # Find all clickable elements (links, buttons, etc.)
                clickable_elements = page.query_selector_all('a, button, [role="button"], [role="link"]')
                
                for element in clickable_elements:
                    try:
                        # Get href for links
                        href = element.get_attribute('href')
                        if not href:
                            continue
                            
                        # Get text content
                        link_text = element.text_content().strip()
                        
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
                
                # Get page content
                main_content = soup.find('main')
                if not main_content:
                    main_content = soup.find('article')
                if not main_content:
                    main_content = soup.find('body')
                
                title = page.title()
                content = main_content.get_text(separator=' ', strip=True) if main_content else ""
                
                print(f"[DEBUG] Found {len(page_links)} links on sign-in page")
                page.close()
                return {
                    "url": url,
                    "title": title,
                    "content": content,
                    "type": "sign-in",
                    "links": page_links
                }
            
            # For other pages, use regular requests
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            main_content = soup.find('main')
            if not main_content:
                main_content = soup.find('article')
            if not main_content:
                main_content = soup.find('body')
            title = soup.title.string if soup.title else ""
            content = main_content.get_text(separator=' ', strip=True) if main_content else ""
            page_links = []
            if len(content) > 100:  # Minimum content length threshold
                return {
                    "url": url,
                    "title": title,
                    "content": content,
                    "type": "content",
                    "links": page_links
                }
            return None
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    def crawl_va_pages(self, start_url: str, max_pages: int = 100) -> List[Dict]:
        """
        Crawl VA.gov starting from a specific URL
        """
        pages_data = []
        to_visit = [start_url]
        
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
                print(f"Successfully scraped: {current_url}")
                
                # If this is a resource page, add its article links to the queue
                if page_data.get('type') == 'resource':
                    for link in page_data.get('links', []):
                        if link['url'] not in self.visited_urls and link['url'] not in to_visit:
                            to_visit.append(link['url'])
            
            try:
                response = requests.get(current_url, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    
                    # Convert relative URLs to absolute
                    if href.startswith('/'):
                        full_url = self.base_url + href
                    elif href.startswith(self.base_url):
                        full_url = href
                    else:
                        continue
                    
                    # Clean and validate URL
                    full_url = self.clean_url(full_url)
                    if self.is_valid_url(full_url) and full_url not in self.visited_urls and full_url not in to_visit:
                        to_visit.append(full_url)
                        
            except Exception as e:
                print(f"Error processing links from {current_url}: {str(e)}")
                
        return pages_data

    def save_to_json(self, data: List[Dict], filename: str):
        """
        Save scraped data to a JSON file
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    scraper = VAScraper()
    
    # Only focus on the sign-in page
    start_urls = [
        "https://www.va.gov/sign-in/"
    ]
    
    all_pages_data = []
    for url in start_urls:
        print(f"\nStarting crawl from: {url}")
        pages_data = scraper.crawl_va_pages(url, max_pages=100)
        all_pages_data.extend(pages_data)
        print(f"Completed crawl from {url}. Total pages so far: {len(all_pages_data)}")
    
    print(f"\nScraping completed. Total pages scraped: {len(all_pages_data)}")
    scraper.save_to_json(all_pages_data, "va_content.json") 