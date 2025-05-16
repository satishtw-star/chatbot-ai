import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import json
import os
import time
import urllib.parse

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
        Scrape content from a VA.gov page
        """
        try:
            # Add delay to be respectful to VA.gov servers
            time.sleep(1)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style, and nav elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
                
            # Get main content
            main_content = soup.find('main')
            if not main_content:
                main_content = soup.find('article')
            if not main_content:
                main_content = soup.find('body')
                
            title = soup.title.string if soup.title else ""
            content = main_content.get_text(separator=' ', strip=True) if main_content else ""
            
            # Only return if we have meaningful content
            if len(content) > 100:  # Minimum content length threshold
                return {
                    "url": url,
                    "title": title,
                    "content": content
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
    
    # Important VA.gov sections to start from
    start_urls = [
        "https://www.va.gov/",
        "https://www.va.gov/health-care/",
        "https://www.va.gov/benefits/",
        "https://www.va.gov/education/",
        "https://www.va.gov/disability/",
        "https://www.va.gov/pension/",
        "https://www.va.gov/housing-assistance/",
        "https://www.va.gov/life-insurance/",
        "https://www.va.gov/careers-employment/",
        "https://www.va.gov/records/",
    ]
    
    all_pages_data = []
    for url in start_urls:
        print(f"\nStarting crawl from: {url}")
        pages_data = scraper.crawl_va_pages(url, max_pages=100)  # Increased to 100 pages per section
        all_pages_data.extend(pages_data)
        print(f"Completed crawl from {url}. Total pages so far: {len(all_pages_data)}")
    
    print(f"\nScraping completed. Total pages scraped: {len(all_pages_data)}")
    scraper.save_to_json(all_pages_data, "va_content.json") 