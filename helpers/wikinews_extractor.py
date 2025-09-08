import re
import time
import textwrap
import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import trafilatura
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

TEXT_MIN_LEN_FOR_NEWSPAPER = 100
SELENIUM_WAIT_SECS = 5

WRAP = lambda t: textwrap.fill(t or "", width=100)


# Utilities
def word_counter(text):
    return len(re.findall(r"\b\w+\b", text or ""))


def basic_text_cleaner(s):
    if not s:
        return ""
    s = s.replace("\n", " ")
    s = s.replace("\\", "")
    s = (s
         .replace("“", "'")
         .replace("”", "'")
         .replace("’", "'")
         .replace("\"", "'"))

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def bin_by_length(word_count):
    if word_count < 250:
        return "Short"
    elif word_count < 500:
        return "Medium"
    else:
        return "Long"


# Wikinews scraping
def extract_wikinews_article_and_sources(wikinews_url):
    """
    Returns (source_links, article_text)
    """
    r = requests.get(wikinews_url, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        return [], ""

    soup = BeautifulSoup(r.content, "html.parser")
    content_div = soup.find("div", class_="mw-parser-output")
    if not content_div:
        return [], ""

    # Only top-level <p> in content area
    paragraphs = content_div.find_all("p", recursive=False)
    article_text = "\n".join(
        p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
    )

    # Grab 'Sources' section links
    source_links = []
    # Many Wikinews pages use an h2 with span id="Sources"
    sources_h2 = content_div.find(lambda tag: tag.name in ["h2", "h3"] and tag.find(id="Sources"))
    if sources_h2:
        # Find the next UL after this heading
        nxt = sources_h2.find_next_sibling()
        while nxt is not None and nxt.name not in ["ul", "ol"]:
            nxt = nxt.find_next_sibling()
        if nxt and nxt.name in ["ul", "ol"]:
            for a in nxt.find_all("a", class_="external text", href=True):
                source_links.append(a["href"])

    return source_links, article_text


# Source extraction
def extract_with_newspaper(url):
    art = Article(url)
    art.download()
    art.parse()
    title = art.title.strip() if art.title else url
    text = art.text.strip() if art.text else ""
    if len(text) < TEXT_MIN_LEN_FOR_NEWSPAPER:
        raise ValueError("newspaper3k text too short; fallback")
    return {"title": title, "text": text}


def extract_with_selenium(url):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        time.sleep(SELENIUM_WAIT_SECS)
        html = driver.page_source
    finally:
        driver.quit()

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.text.strip() if soup.title else url
    text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    return {"title": title, "text": text.strip()}


def extract_source_article_content(url):
    """
    Best-effort extractor with graceful fallbacks.
    Returns dict {"title": str, "text": str}
    """
    # Try newspaper3k
    try:
        return extract_with_newspaper(url)
    except Exception:
        pass

    # Try direct requests + trafilatura
    try:
        r = requests.get(url, headers=HEADERS, timeout=25)
        if r.status_code == 200:
            html = r.text
            title = BeautifulSoup(html, "html.parser").title
            title_text = (title.text.strip() if title else url)
            text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
            if len(text.strip()) >= 50:
                return {"title": title_text, "text": text.strip()}
    except Exception:
        pass

    # Selenium fallback if enabled
    try:
        return extract_with_selenium(url)
    except Exception as e:
        # Final fallback: return minimal stub
        return {"title": url, "text": f"[Extraction failed: {e}]".strip()}



# Main pipeline
def process_wikinews_items(items):
    """
    items: list of dicts with keys {'date','title','url'}
    Returns: list of records with cleaned fields + counts + bins + sources and reference extracts
    """
    records = []

    for item in items:
        print(f"\n=== Processing: {item['title']} ===")
        source_links, article_text = extract_wikinews_article_and_sources(item["url"])
        article_text = basic_text_cleaner(article_text)

        # Pull each linked source
        reference_data = []
        for link in source_links:
            res = extract_source_article_content(link)
            res_clean = {
                "title": basic_text_cleaner(res.get("title", "")),
                "text": basic_text_cleaner(res.get("text", "")),
                "url": link
            }
            reference_data.append(res_clean)

        # Join all source texts into one "source_data" string (for your embedding pipeline)
        source_data_text = " ".join(ref["text"] for ref in reference_data if ref.get("text"))

        # Counts & bins
        wc_article = word_counter(article_text)
        wc_source = word_counter(source_data_text)
        wc_bin = bin_by_length(wc_article)

        rec = {
            "date": item["date"],
            "title": item["title"],
            "url": item["url"],
            "category": item["category"],
            "article_text": article_text,
            "source_links": source_links,
            "reference_data": reference_data,   # structured list (kept JSON-serializable)
            "source_data": source_data_text,
            "word_count": wc_article,
            "word_count_bin": wc_bin,
            "word_count_source_data": wc_source,
        }
        records.append(rec)

    return records


# Entry point
if __name__ == "__main__":
    # Provide your Wikinews items here (extend as needed)
    wikinews_items = [
        {'date': '2022-06-03', 'title': "Scientists discover seagrass off Australia is world's largest plant",
         'url': "https://en.wikinews.org/wiki/Scientists_discover_seagrass_off_Australia_is_world%27s_largest_plant",
         'category': 'Science & Environment'},
        {'date': '2022-09-26', 'title': 'United Kingdom buries Queen Elizabeth II after state funeral',
         'url': 'https://en.wikinews.org/wiki/United_Kingdom_buries_Queen_Elizabeth_II_after_state_funeral',
         'category': 'Politics & Policy'},
        {'date': '2023-08-13', 'title': 'US: Tulsa residents approve $814 million infrastructure package',
         'url': 'https://en.wikinews.org/wiki/US%3A_Tulsa_residents_approve_%24814_million_infrastructure_package',
         'category': 'Politics & Policy'},
        {'date': '2025-02-10',
         'title': 'UK heavy metal band Black Sabbath announces final performance with original lineup',
         'url': 'https://en.wikinews.org/wiki/UK_heavy_metal_band_Black_Sabbath_announces_final_performance_with_original_lineup',
         'category': 'Entertainment'},
        {'date': '2025-02-19', 'title': '78th British Academy Film Awards held in London',
         'url': 'https://en.wikinews.org/wiki/78th_British_Academy_Film_Awards_held_in_London',
         'category': 'Entertainment'},
        {'date': '2025-03-10', 'title': 'India defeats New Zealand to win 2025 Champions Trophy',
         'url': 'https://en.wikinews.org/wiki/India_defeats_New_Zealand_to_win_2025_Champions_Trophy',
         'category': 'Sports'},
        {'date': '2025-04-20', 'title': 'Ryan Gosling cast in upcoming Star Wars film',
         'url': 'https://en.wikinews.org/wiki/Ryan_Gosling_cast_in_upcoming_Star_Wars_film',
         'category': 'Entertainment'},
        {'date': '2025-04-23', 'title': 'Researchers film colossal squid in its natural habitat for the first time',
         'url': 'https://en.wikinews.org/wiki/Researchers_film_colossal_squid_in_its_natural_habitat_for_the_first_time',
         'category': 'Science & Environment'},
        {'date': '2025-05-16', 'title': 'Thai officials seize 238 tons of illegal e-waste at Bangkok port',
         'url': 'https://en.wikinews.org/wiki/Thai_officials_seize_238_tons_of_illegal_e-waste_at_Bangkok_port',
         'category': 'Science & Environment'},
        {'date': '2025-07-10', 'title': '20-year-old astrophotographer captures rare solar eclipse on Saturn',
         'url': 'https://en.wikinews.org/wiki/20-year-old_astrophotographer_captures_rare_solar_eclipse_on_Saturn',
         'category': 'Science & Environment'},
        {'date': '2023-05-01', 'title': 'Microsoft, Nware sign 10-year cloud gaming deal',
         'url': 'https://en.wikinews.org/wiki/Microsoft,_Nware_sign_10-year_cloud_gaming_deal',
         'category': 'Business & Technology'},
        {'date': '2018-06-30', 'title': 'FIFA World Cup 2018 Last 16: France, Uruguay send Argentina, Portugal home',
         'url': 'https://en.wikinews.org/wiki/FIFA_World_Cup_2018_Last_16:_France,_Uruguay_send_Argentina,_Portugal_home',
         'category': 'Sports'},
        {'date': '2021-07-14', 'title': 'European Union to reduce carbon emissions by 55% of 1990 levels by 2030',
         'url': 'https://en.wikinews.org/wiki/European_Union_to_reduce_carbon_emissions_by_55%_of_1990_levels_by_2030',
         'category': 'Politics & Policy'},
        {'date': '2024-11-15', 'title': 'SpaceX will return stranded astronauts in February 2025, NASA announces',
         'url': 'https://en.wikinews.org/wiki/SpaceX_will_return_stranded_astronauts_in_February_2025,_NASA_announces',
         'category': 'Business & Technology'},
        {'date': '2023-12-17',
         'title': 'GSK rejects three Unilever bids to buy consumer healthcare arm, says unit was fundamentally undervalued',
         'url': 'https://en.wikinews.org/wiki/GSK_rejects_three_Unilever_bids_to_buy_consumer_healthcare_arm,_says_%22fundamentally_undervalued%22',
         'category': 'Business & Technology'}
    ]

    # Run pipeline
    new_records = process_wikinews_items(wikinews_items)
    df = pd.DataFrame(new_records)
    df.to_csv("wikinews_data.csv", index=False)
    df.to_json("wikinews_data.json", orient="records", indent=3, ensure_ascii=False)

    # preview a compact log
    print("\n--- Preview ---")
    for r in new_records:
        print(f"* {r['title']}: article_wc={r['word_count']}, sources={len(r['source_links'])}, "
              f"source_wc={r['word_count_source_data']}")
