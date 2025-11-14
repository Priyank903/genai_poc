"""
Web search tool for Clinical Agent
Uses DuckDuckGo search for queries outside reference materials
"""
import logging
from typing import List, Dict
from duckduckgo_search import DDGS

logger = logging.getLogger("web_search")

class WebSearchTool:
    def __init__(self):
        self.ddgs = DDGS()
        logger.info("WebSearchTool initialized")
    
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search the web for medical information
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with title, snippet, and URL
        """
        try:
            logger.info(f"Web search initiated for query: {query}")
            results = []
            
            # Use DuckDuckGo search
            search_results = list(self.ddgs.text(query, max_results=max_results))
            
            for i, result in enumerate(search_results):
                results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("body", ""),
                    "url": result.get("href", ""),
                    "rank": i + 1
                })
            
            logger.info(f"Web search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return [{
                "title": "Search Error",
                "snippet": f"Unable to perform web search: {str(e)}",
                "url": "",
                "rank": 0
            }]
    
    def format_search_results(self, results: List[Dict]) -> str:
        """Format search results for LLM context"""
        if not results:
            return "No web search results found."
        
        formatted = "Web Search Results:\n\n"
        for result in results:
            formatted += f"[{result['rank']}] {result['title']}\n"
            formatted += f"   {result['snippet']}\n"
            formatted += f"   Source: {result['url']}\n\n"
        
        return formatted

if __name__ == "__main__":
    # Test the web search tool
    tool = WebSearchTool()
    results = tool.search("SGLT2 inhibitors for kidney disease 2024")
    print(tool.format_search_results(results))

