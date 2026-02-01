"""
Authentic.AI - Web Scraper
Extract article content from URLs for fake news analysis
"""

import logging
import requests
from typing import Dict, Optional
from urllib.parse import urlparse
import re

logger = logging.getLogger(__name__)

# Try to import BeautifulSoup
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("BeautifulSoup not available - install with: pip install beautifulsoup4")
    # Define placeholder for type hints
    from typing import Any
    BeautifulSoup = Any


class WebScraper:
    """
    Extract article content from web pages
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        self.timeout = 10
    
    def extract_article(self, url: str) -> Dict:
        """
        Extract article content from URL
        
        Args:
            url: Article URL
            
        Returns:
            Dict with title, text, author, date, etc.
        """
        result = {
            "success": False,
            "url": url,
            "title": None,
            "text": None,
            "author": None,
            "published_date": None,
            "publisher": None,
            "description": None,
            "image_url": None,
            "error": None
        }
        
        if not BS4_AVAILABLE:
            result["error"] = "BeautifulSoup not installed"
            return result
        
        try:
            # Fetch the page
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract metadata
            result["title"] = self._extract_title(soup)
            result["description"] = self._extract_description(soup)
            result["author"] = self._extract_author(soup)
            result["published_date"] = self._extract_date(soup)
            result["image_url"] = self._extract_image(soup)
            result["publisher"] = self._extract_publisher(soup, url)
            
            # Extract main content
            result["text"] = self._extract_main_content(soup)
            
            result["success"] = True
            
        except requests.Timeout:
            result["error"] = "Request timed out"
        except requests.RequestException as e:
            result["error"] = f"Failed to fetch URL: {str(e)}"
        except Exception as e:
            result["error"] = f"Extraction error: {str(e)}"
            logger.error(f"Web scraper error: {e}")
        
        return result
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article title"""
        # Try Open Graph title first
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content'].strip()
        
        # Try Twitter title
        tw_title = soup.find('meta', attrs={'name': 'twitter:title'})
        if tw_title and tw_title.get('content'):
            return tw_title['content'].strip()
        
        # Try h1
        h1 = soup.find('h1')
        if h1:
            return h1.get_text().strip()
        
        # Fallback to title tag
        title = soup.find('title')
        if title:
            return title.get_text().strip()
        
        return None
    
    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article description/summary"""
        # Try Open Graph description
        og_desc = soup.find('meta', property='og:description')
        if og_desc and og_desc.get('content'):
            return og_desc['content'].strip()
        
        # Try meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc['content'].strip()
        
        return None
    
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article author"""
        # Try meta author
        meta_author = soup.find('meta', attrs={'name': 'author'})
        if meta_author and meta_author.get('content'):
            return meta_author['content'].strip()
        
        # Try article:author
        article_author = soup.find('meta', property='article:author')
        if article_author and article_author.get('content'):
            return article_author['content'].strip()
        
        # Try common author class patterns
        author_patterns = ['author', 'byline', 'writer', 'contributor']
        for pattern in author_patterns:
            author_elem = soup.find(class_=re.compile(pattern, re.I))
            if author_elem:
                text = author_elem.get_text().strip()
                # Clean up "By " prefix
                text = re.sub(r'^by\s+', '', text, flags=re.I)
                if len(text) < 100:  # Sanity check
                    return text
        
        return None
    
    def _extract_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract publication date"""
        # Try article:published_time
        pub_time = soup.find('meta', property='article:published_time')
        if pub_time and pub_time.get('content'):
            return pub_time['content'].strip()
        
        # Try datePublished in JSON-LD
        scripts = soup.find_all('script', type='application/ld+json')
        for script in scripts:
            try:
                import json
                data = json.loads(script.string)
                if isinstance(data, dict):
                    if 'datePublished' in data:
                        return data['datePublished']
            except:
                pass
        
        # Try time element
        time_elem = soup.find('time')
        if time_elem:
            return time_elem.get('datetime') or time_elem.get_text().strip()
        
        return None
    
    def _extract_image(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract main article image"""
        # Try Open Graph image
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            return og_image['content']
        
        # Try Twitter image
        tw_image = soup.find('meta', attrs={'name': 'twitter:image'})
        if tw_image and tw_image.get('content'):
            return tw_image['content']
        
        return None
    
    def _extract_publisher(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Extract publisher name"""
        # Try Open Graph site_name
        og_site = soup.find('meta', property='og:site_name')
        if og_site and og_site.get('content'):
            return og_site['content'].strip()
        
        # Fallback to domain
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract main article text content"""
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'iframe']):
            element.decompose()
        
        # Try article tag
        article = soup.find('article')
        if article:
            paragraphs = article.find_all('p')
            if paragraphs:
                text = '\n\n'.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50)
                if len(text) > 200:
                    return text
        
        # Try common content classes
        content_patterns = ['article-body', 'article-content', 'post-content', 'entry-content', 'story-body', 'content-body']
        for pattern in content_patterns:
            content = soup.find(class_=re.compile(pattern, re.I))
            if content:
                paragraphs = content.find_all('p')
                if paragraphs:
                    text = '\n\n'.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50)
                    if len(text) > 200:
                        return text
        
        # Fallback: Get all paragraphs with substantial content
        paragraphs = soup.find_all('p')
        substantial = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 100]
        if substantial:
            return '\n\n'.join(substantial[:20])  # Limit to first 20 paragraphs
        
        return None


# Initialize scraper
web_scraper = WebScraper()
