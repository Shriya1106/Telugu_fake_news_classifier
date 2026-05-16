"""
Telugu News Scraper — Sakshi, Eenadu, TV9 Telugu RSS Feeds
==========================================================
Collects real Telugu news headlines to use as "Real" class samples.

Usage:
    python -m src.scraper          # prints scraped headlines
    from src.scraper import scrape_telugu_news
"""

import time
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from .preprocess import clean_telugu_text, is_telugu as is_telugu_content

# RSS feed URLs for major Telugu news outlets
TELUGU_RSS_FEEDS = {
    "Sakshi": "https://www.sakshi.com/rss/telangana",
    "Eenadu": "https://www.eenadu.net/telangana/rss",
    "TV9 Telugu": "https://tv9telugu.com/feed",
}




def scrape_telugu_news(max_per_source: int = 20, timeout: int = 60, max_retries: int = 3) -> list[dict]:
    """
    Scrapes Telugu news headlines from RSS feeds with retry logic and timeout handling.

    Args:
        max_per_source: Maximum articles to scrape per source (default: 20)
        timeout: Total timeout in seconds for all scraping operations (default: 60)
        max_retries: Number of retry attempts for failed requests (default: 3)

    Returns:
        list of dict with keys: 'text', 'source', 'label'
        label is always 0 (Real) since these come from verified news outlets.
    """
    articles = []
    start_time = time.time()

    for source_name, feed_url in TELUGU_RSS_FEEDS.items():
        # Check if we've exceeded the total timeout
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            print(f"⏱️  Timeout reached ({timeout}s), stopping scraper")
            break
        
        # Retry logic for each source
        for attempt in range(1, max_retries + 1):
            try:
                # Check timeout before each attempt
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    print(f"⏱️  Timeout reached ({timeout}s), stopping scraper")
                    break
                
                feed = feedparser.parse(feed_url)
                
                # Check if feed was successfully parsed
                if not feed.entries:
                    raise Exception("No entries found in feed")
                
                scraped_count = 0
                for entry in feed.entries[:max_per_source]:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    if summary:
                        summary = BeautifulSoup(summary, "html.parser").get_text(separator=" ")
                    raw_text = f"{title}. {summary}" if summary else title
                    
                    # Filter non-Telugu content using Unicode detection
                    if not is_telugu_content(raw_text):
                        continue
                    
                    cleaned = clean_telugu_text(raw_text)

                    if cleaned and len(cleaned) > 10:
                        articles.append({
                            "text": cleaned,
                            "source": source_name,
                            "label": 0,  # Real — verified news source
                        })
                        scraped_count += 1
                
                print(f"✅ {source_name}: scraped {scraped_count} Telugu articles (attempt {attempt})")
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt < max_retries:
                    print(f"⚠️  {source_name}: attempt {attempt} failed — {e}. Retrying...")
                    time.sleep(1)  # Brief delay before retry
                else:
                    print(f"❌ {source_name}: all {max_retries} attempts failed — {e}")

    return articles


if __name__ == "__main__":
    results = scrape_telugu_news()
    print(f"\nTotal articles scraped: {len(results)}")
    for r in results[:5]:
        print(f"  [{r['source']}] {r['text'][:80]}…")
    
    # Save results to parquet file
    if results:
        df = pd.DataFrame(results)
        output_path = "telugu_news.parquet"
        df.to_parquet(output_path, index=False)
        print(f"\n✅ Saved {len(results)} articles to {output_path}")
