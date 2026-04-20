import requests
from bs4 import BeautifulSoup

COMMON_RSS_PATHS = [
    "/feed",
    "/rss",
    "/feed.xml",
    "/rss.xml",
    "/atom.xml",
    "/feeds/posts/default",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RSSChecker/1.0)"}


def check_rss(url):
    print(f"\nChecking {url}...")

    for path in COMMON_RSS_PATHS:
        full_url = url.rstrip("/") + path
        try:
            r = requests.get(full_url, timeout=5, headers=HEADERS)
            content_type = r.headers.get("Content-Type", "")

            # Check content type AND body content
            is_xml = any(ct in content_type for ct in ["rss", "xml", "atom"])
            has_rss_content = "<rss" in r.text or "<feed" in r.text

            if r.status_code == 200 and (is_xml or has_rss_content):
                print(f"Found RSS at: {full_url}")
                return full_url

        except Exception as e:
            continue

    # Try detecting from homepage <link> tags
    try:
        r = requests.get(url, timeout=5, headers=HEADERS)
        soup = BeautifulSoup(r.text, "html.parser")
        feeds = soup.find_all(
            "link", type=["application/rss+xml", "application/atom+xml"]
        )
        if feeds:
            feed_url = feeds[0].get("href")
            print(f"Found RSS in page head: {feed_url}")
            return feed_url
    except Exception as e:
        print(f" Could not scan homepage: {e}")

    print(f"No RSS found for: {url}")
    return None


SITES = [
    "https://fakti.bg/",
    "https://novini.bg/",
    "https://24chasa.bg",
    "https://nova.bg/news",
    "https://www.dnevnik.bg",
    "https://www.capital.bg",
    "https://dnevnik.bg",
    "https://blitz.bg",
    "https://mediapool.bg",
    "https://capital.bg",
    "https://www.actualno.com/",
    "https://www.dnes.bg/",
]

print("=" * 50)
print("RSS Feed Checker")
print("=" * 50)

results = {}
for site in SITES:
    rss_url = check_rss(site)
    results[site] = rss_url

print("\n" + "=" * 50)
print("Summary")
print("=" * 50)
for site, rss in results.items():
    status = f"{rss}" if rss else "No RSS found"
    print(f"{site:<30} {status}")
