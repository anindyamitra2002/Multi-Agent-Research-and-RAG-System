import os
import sys
import asyncio
from typing import List, Dict, Optional, Set
from urllib.parse import urlparse
from langchain_community.tools import DuckDuckGoSearchResults, TavilySearchResults
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter, BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from dotenv import load_dotenv

load_dotenv()

class DeepWebCrawler:
    def __init__(self, 
                 max_search_results: int = 5,
                 max_external_links: int = 3,
                 word_count_threshold: int = 50,
                 content_filter_type: str = 'pruning',
                 filter_threshold: float = 0.48):
        """
        Initialize the Deep Web Crawler with support for one-level deep crawling
        
        Args:
            max_search_results (int): Maximum number of search results to process
            max_external_links (int): Maximum number of external links to crawl per page
            word_count_threshold (int): Minimum word count for crawled content
            content_filter_type (str): Type of content filter ('pruning' or 'bm25')
            filter_threshold (float): Threshold for content filtering
        """
        self.max_search_results = max_search_results
        self.max_external_links = max_external_links
        self.word_count_threshold = word_count_threshold
        self.content_filter_type = content_filter_type
        self.filter_threshold = filter_threshold
        self.crawled_urls: Set[str] = set()

    def _create_web_search_tool(self):
        return TavilySearchResults(max_results=self.max_search_results)

    def _create_content_filter(self, user_query: Optional[str] = None):
        if self.content_filter_type == 'bm25' and user_query:
            return BM25ContentFilter(
                user_query=user_query, 
                bm25_threshold=self.filter_threshold
            )
        return PruningContentFilter(
            threshold=self.filter_threshold,
            threshold_type="fixed",
            min_word_threshold=self.word_count_threshold
        )

    def _extract_links_from_search_results(self, results: List[Dict]) -> List[str]:
        """Safely extract URLs from search results"""
        urls = []
        for result in results:
            if isinstance(result, dict) and 'url' in result:
                urls.append(result['url'])
            elif isinstance(result, str):
                urls.append(result)
        return urls

    def _extract_url_from_link(self, link):
        """Extract URL string from link object which might be a dict or string"""
        if isinstance(link, dict):
            return link.get('url', '')  # Assuming the URL is stored in a 'url' key
        elif isinstance(link, str):
            return link
        return ''
    
    def _process_crawl_result(self, result) -> Dict:
        """Process individual crawl result into structured format"""
        return {
            "url": result.url,
            "success": result.success,
            "title": result.metadata.get('title', 'N/A'),
            "content": result.markdown_v2.raw_markdown if result.success else result.error_message,
            "word_count": len(result.markdown_v2.raw_markdown.split()) if result.success else 0,
            "links": {
                "internal": result.links.get('internal', []),
                "external": result.links.get('external', [])
            },
            "images": len(result.media.get('images', []))
        }

    async def crawl_urls(self, urls: List[str], user_query: Optional[str] = None, depth: int = 0):
        """
        Crawl URLs with support for external link crawling
        
        Args:
            urls (List[str]): List of URLs to crawl
            user_query (Optional[str]): Query for content filtering
            depth (int): Current crawl depth (0 for initial, 1 for external links)
        
        Returns:
            List of crawl results including external link content
        """
        if not urls or depth > 1:
            return []

        # Filter out already crawled URLs
        new_urls = [url for url in urls if url not in self.crawled_urls]
        if not new_urls:
            return []

        async with AsyncWebCrawler(
            browser_type="chromium",
            headless=True,
            verbose=True
        ) as crawler:
            content_filter = self._create_content_filter(user_query)
            
            results = await crawler.arun_many(
                urls=new_urls,
                word_count_threshold=self.word_count_threshold,
                cache_mode=CacheMode.BYPASS,
                markdown_generator=DefaultMarkdownGenerator(content_filter=content_filter),
                exclude_external_links=True,
                exclude_social_media_links=True,
                remove_overlay_elements=True,
                simulate_user=True,
                magic=True
            )

            processed_results = []
            external_urls = set()

            # Process results and collect external URLs
            for result in results:
                self.crawled_urls.add(result.url)
                processed_result = self._process_crawl_result(result)
                processed_results.append(processed_result)

                if depth == 0 and result.success:
                    # Collect unique external URLs for further crawling
                    external_links = result.links.get('external', [])[:self.max_external_links]
                    external_urls.update(
                        self._extract_url_from_link(link) 
                        for link in external_links 
                        if self._extract_url_from_link(link) 
                        and self._extract_url_from_link(link) not in self.crawled_urls
                    )

            # Crawl external links if at depth 0
            if depth == 0 and external_urls and False:
                external_results = await self.crawl_urls(
                    list(external_urls),
                    user_query=user_query,
                    depth=0
                )
                processed_results.extend(external_results)

            return processed_results

    async def search_and_crawl(self, query: str) -> List[Dict]:
        """
        Perform web search and deep crawl of results
        
        Args:
            query (str): Search query
        
        Returns:
            List of crawled content results including external links
        """

        search_tool = self._create_web_search_tool()
        search_results = search_tool.invoke(query)
        
        # Handle different types of search results
        if isinstance(search_results, str):
            urls = [search_results]
        elif isinstance(search_results, list):
            urls = self._extract_links_from_search_results(search_results)
        else:
            print(f"Unexpected search results format: {type(search_results)}")
            return []
        
        if not urls:
            print("No valid URLs found in search results")
            return []
        
        print(f"Initial search found {len(urls)} URLs for query: {query}")
        print(urls)
        crawl_results = await self.crawl_urls(urls, user_query=query)
        
        return crawl_results


class ResourceCollectionAgent:
    def __init__(self, max_results_per_query: int = 10):
        """
        Initialize the Resource Collection Agent
        
        Args:
            max_results_per_query (int): Maximum number of results per search query
        """
        self.max_results_per_query = max_results_per_query
        self.search_tool = TavilySearchResults(max_results=max_results_per_query)

    def _is_valid_domain(self, url: str, valid_domains: List[str]) -> bool:
        """Check if URL belongs to allowed domains"""
        try:
            domain = urlparse(url).netloc.lower()
            return any(valid_domain in domain for valid_domain in valid_domains)
        except:
            return False

    def _extract_search_result(self, result) -> Optional[Dict]:
        """Safely extract information from a search result"""
        try:
            if isinstance(result, dict):
                return {
                    "title": result.get("title", "No title"),
                    "url": result.get("url", ""),
                    "snippet": result.get("snippet", "No description")
                }
            elif isinstance(result, str):
                return {
                    "title": "Unknown",
                    "url": result,
                    "snippet": "No description available"
                }
            return None
        except Exception as e:
            print(f"Error processing search result: {str(e)}")
            return None

    async def collect_resources(self) -> Dict[str, List[Dict]]:
        """
        Collect AI/ML resources from specific platforms
        
        Returns:
            Dictionary with categorized resource links
        """
        search_queries = {
            "datasets": [
                ("kaggle", "site:kaggle.com/datasets machine learning"),
                ("huggingface", "site:huggingface.co/datasets artificial intelligence")
            ],
            "repositories": [
                ("github", "site:github.com AI tools repository")
            ]
        }

        results = {
            "kaggle_datasets": [],
            "huggingface_datasets": [],
            "github_repositories": []
        }

        for category, queries in search_queries.items():
            for platform, query in queries:
                try:
                    search_results = self.search_tool.invoke(query)
                    
                    # Handle different result formats
                    if isinstance(search_results, str):
                        search_results = [search_results]
                    elif not isinstance(search_results, list):
                        print(f"Unexpected search results format for {platform}: {type(search_results)}")
                        continue
                    
                    # Filter results based on domain
                    valid_domains = {
                        "kaggle": ["kaggle.com"],
                        "huggingface": ["huggingface.co"],
                        "github": ["github.com"]
                    }
                    
                    for result in search_results:
                        processed_result = self._extract_search_result(result)
                        if processed_result and self._is_valid_domain(
                            processed_result["url"], 
                            valid_domains[platform]
                        ):
                            if platform == "kaggle":
                                results["kaggle_datasets"].append(processed_result)
                            elif platform == "huggingface":
                                results["huggingface_datasets"].append(processed_result)
                            elif platform == "github":
                                results["github_repositories"].append(processed_result)
                    
                except Exception as e:
                    print(f"Error collecting {platform} resources: {str(e)}")
                    continue

        return results

def main():
    async def run_examples():
        # Test DeepWebCrawler
        deep_crawler = DeepWebCrawler(
            max_search_results=3,
            max_external_links=2,
            word_count_threshold=50
        )
        
        crawl_results = await deep_crawler.search_and_crawl(
            "Adani Defence & Aerospace"
        )
        
        print("\nDeep Crawler Results:")
        for result in crawl_results:
            print(f"URL: {result['url']}")
            print(f"Title: {result['title']}")
            print(f"Word Count: {result['word_count']}")
            print(f"External Links: {len(result['links']['external'])}\n")

        # Test ResourceCollectionAgent
        resource_agent = ResourceCollectionAgent(max_results_per_query=5)
        resources = await resource_agent.collect_resources()
        
        print("\nResource Collection Results:")
        for category, items in resources.items():
            print(f"\n{category.upper()}:")
            for item in items:
                print(f"Title: {item['title']}")
                print(f"URL: {item['url']}")
                print("---")

    asyncio.run(run_examples())

if __name__ == "__main__":
    main()