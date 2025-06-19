import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import os
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
from datetime import datetime

# Configuration
MAX_PAGES = 1000
MAX_WORKERS = 10
CACHE_DIR = "cache"
REQUEST_TIMEOUT = 15
DELAY_BETWEEN_REQUESTS = 0.5
BASE_URL = "https://apps.odoo.com"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("scraper.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

def get_session():
    """Create a session with retry capability"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    })
    return session

def clean_text(text):
    """Clean and normalize text"""
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    return text

def clean_price(price_str):
    """Robust price cleaning"""
    price_str = clean_text(price_str)
    if not price_str or price_str.lower() in ('free', 'n/a', ''):
        return 0.0
    try:
        # Handle price formats like €100.00 or $100.00
        clean_str = re.sub(r'[^\d.,]', '', price_str)
        clean_str = clean_str.replace(',', '')
        return float(clean_str)
    except:
        return 0.0

def clean_purchases(purchase_str):
    """Clean purchase numbers"""
    purchase_str = clean_text(purchase_str)
    if not purchase_str:
        return 0
    try:
        return int(''.join(c for c in purchase_str if c.isdigit()) or 0)
    except:
        return 0

def clean_rating(rating_str):
    """Clean rating strings"""
    rating_str = clean_text(rating_str)
    if not rating_str or rating_str.lower() == "n/a":
        return 0.0
    try:
        # Extract just the numeric part from strings like "4.5/5.0"
        match = re.search(r'(\d+\.?\d*)', rating_str)
        if match:
            return float(match.group(1))
        return float(rating_str)
    except:
        return 0.0

def extract_version_support(text):
    """Extract supported Odoo versions from text"""
    versions = []
    if not text:
        return versions
    
    # Match patterns like "Odoo 14.0, 15.0" or "14.0-16.0"
    matches = re.findall(r'(?:Odoo\s*)?(\d+\.\d+)', text)
    for match in matches:
        if match not in versions:
            versions.append(match)
    
    return versions

def get_cache_filename(page_num):
    """Generate cache filename for a page"""
    return os.path.join(CACHE_DIR, f"page_{page_num}.json")

def is_cached(page_num):
    """Check if page results are cached"""
    cache_file = get_cache_filename(page_num)
    return os.path.exists(cache_file)

def get_from_cache(page_num):
    """Get data from cache"""
    cache_file = get_cache_filename(page_num)
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading cache for page {page_num}: {e}")
        return None

def save_to_cache(page_num, data):
    """Save data to cache"""
    cache_file = get_cache_filename(page_num)
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving cache for page {page_num}: {e}")

def extract_text_safely(element, selector, default=""):
    """Safely extract text from an element"""
    try:
        found = element.select_one(selector)
        return clean_text(found.get_text()) if found else default
    except:
        return default

def extract_attr_safely(element, selector, attr, default=""):
    """Safely extract attribute from an element"""
    try:
        found = element.select_one(selector)
        return found[attr] if found and attr in found.attrs else default
    except:
        return default

def extract_app_details(app_url, session):
    """Extract additional details from app detail page"""
    details = {
        'description_long': '',
        'version_support': [],
        'last_updated': '',
        'category': '',
        'dependencies': []
    }
    
    try:
        response = session.get(app_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract long description
        details['description_long'] = extract_text_safely(soup, "div.oe_app_description")
        
        # Extract version support
        version_text = extract_text_safely(soup, "div.oe_app_support")
        details['version_support'] = extract_version_support(version_text)
        
        # Extract last updated date
        details['last_updated'] = extract_text_safely(
            soup, 
            "div.oe_app_info:contains('Last Updated')", 
            ''
        ).replace('Last Updated:', '').strip()
        
        # Extract category
        details['category'] = extract_text_safely(
            soup, 
            "div.oe_app_info:contains('Category')", 
            ''
        ).replace('Category:', '').strip()
        
        # Extract dependencies
        deps_section = soup.select_one("div.oe_app_dependencies")
        if deps_section:
            details['dependencies'] = [
                clean_text(li.get_text()) 
                for li in deps_section.select("li")
            ]
            
    except Exception as e:
        logger.warning(f"Error scraping app details from {app_url}: {str(e)}")
    
    return details

def extract_page_data(soup, session):
    """Extract app data from page soup"""
    apps = []
    for card in soup.select("div.loempia_app_card"):
        try:
            # Extract app name
            app_name = extract_text_safely(card, "h5")
            if not app_name or app_name == "Unknown":
                continue  # Skip invalid entries
                
            # Get app URL
            app_url = extract_attr_safely(card, "a", "href")
            if app_url:
                app_url = BASE_URL + app_url
                
            # Extract purchases
            purchases_elem = card.find("span", title=lambda x: x and "Total Purchases" in x)
            purchases = purchases_elem.get_text(strip=True) if purchases_elem else "0"
            
            # Get additional details from app page
            app_details = extract_app_details(app_url, session) if app_url else {}
                
            # Create app dictionary with consistent lowercase column names
            app = {
                "app_name": app_name,
                "vendor": extract_text_safely(card, "div.loempia_panel_author"),
                "price": extract_text_safely(card, "div.loempia_panel_price"),
                "purchases": purchases,
                "rating": extract_attr_safely(card, "span.loempia_rating_stars", "title", "N/A"),
                "url": app_url,
                "short_description": extract_text_safely(card, "p.loempia_panel_summary"),
                "scraped_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **app_details
            }
            apps.append(app)
        except Exception as e:
            logger.debug(f"Error extracting app data: {str(e)}")
            continue
    
    return apps

def scrape_page(page_num, session):
    """Scrape a single page with caching"""
    # Check cache first
    if is_cached(page_num):
        logger.debug(f"Loading page {page_num} from cache")
        return get_from_cache(page_num)
    
    url = f"{BASE_URL}/apps/modules/browse?page={page_num}"
    try:
        logger.info(f"Scraping page {page_num}")
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        apps = extract_page_data(soup, session)
        
        # Save to cache
        save_to_cache(page_num, apps)
        
        return apps
    except Exception as e:
        logger.error(f"Error scraping page {page_num}: {str(e)}")
        return []

def scrape_all_pages(max_pages=MAX_PAGES, max_workers=MAX_WORKERS):
    """Scrape all pages in parallel"""
    all_apps = []
    session = get_session()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all scraping tasks
        future_to_page = {
            executor.submit(scrape_page, page_num, session): page_num 
            for page_num in range(1, max_pages + 1)
        }
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_page), total=max_pages, desc="Scraping pages"):
            page_num = future_to_page[future]
            try:
                apps = future.result()
                all_apps.extend(apps)
                logger.info(f"Page {page_num} completed, found {len(apps)} apps")
                time.sleep(DELAY_BETWEEN_REQUESTS)  # Small delay between submissions
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {str(e)}")
    
    return pd.DataFrame(all_apps)

def clean_dataframe(df):
    """Clean and transform the dataframe with robust column handling"""
    logger.info("Cleaning and transforming data...")
    
    if df.empty:
        return df
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Standardize column names if needed
    column_mapping = {
        'Price': 'price',
        'Purchases': 'purchases',
        'Rating': 'rating',
        'App': 'app_name',
        'Vendor': 'vendor'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Clean and transform columns with fallbacks
    if 'price' in df.columns:
        df["price"] = df["price"].apply(clean_price)
    else:
        logger.warning("'price' column not found in DataFrame")
        df["price"] = 0.0
    
    if 'purchases' in df.columns:
        df["purchases"] = df["purchases"].apply(clean_purchases)
    else:
        logger.warning("'purchases' column not found in DataFrame")
        df["purchases"] = 0
    
    if 'rating' in df.columns:
        df["rating"] = df["rating"].apply(clean_rating)
    else:
        logger.warning("'rating' column not found in DataFrame")
        df["rating"] = 0.0
    
    # Convert last_updated to datetime if exists
    if 'last_updated' in df.columns:
        df['last_updated'] = pd.to_datetime(
            df['last_updated'], 
            errors='coerce', 
            format='%b %d, %Y'
        )
    
    # Calculate days since last update if possible
    if 'last_updated' in df.columns:
        df['days_since_update'] = (pd.to_datetime('today') - df['last_updated']).dt.days
    
    # Add revenue estimate if we have the needed columns
    if all(col in df.columns for col in ['price', 'purchases']):
        df['revenue_estimate'] = df['price'] * df['purchases']
    
    return df

def save_to_csv(df, filename="odoo_apps_full.csv"):
    """Save dataframe to CSV with additional checks"""
    try:
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"Data saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving to CSV: {str(e)}")
        return False

def save_to_excel(df, filename="odoo_apps_full.xlsx"):
    """Save dataframe to Excel with multiple sheets"""
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='All Apps', index=False)
            
            # Summary sheet
            if not df.empty:
                summary_data = {
                    'Total Apps': [len(df)],
                    'Total Vendors': [df['vendor'].nunique() if 'vendor' in df.columns else 0],
                    'Free Apps': [len(df[df['price'] == 0]) if 'price' in df.columns else 0],
                    'Paid Apps': [len(df[df['price'] > 0]) if 'price' in df.columns else 0],
                    'Average Price': [df['price'].mean() if 'price' in df.columns else 0],
                    'Average Rating': [df['rating'].mean() if 'rating' in df.columns else 0],
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"Data saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving to Excel: {str(e)}")
        return False

def main():
    start_time = time.time()
    logger.info("Starting Odoo apps scraper")
    
    # Scrape data
    df = scrape_all_pages(max_pages=MAX_PAGES)
    logger.info(f"Scraping completed. Found {len(df)} apps.")
    
    # Print columns for debugging
    logger.info(f"Columns in raw data: {df.columns.tolist()}")
    
    # Clean data
    df = clean_dataframe(df)
    
    # Save to CSV
    csv_success = save_to_csv(df)
    
    # Save to Excel (optionnel)
    excel_success = save_to_excel(df)
    
    # Print summary
    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
    logger.info(f"Total apps scraped: {len(df)}")
    
    if csv_success:
        logger.info("✅ CSV file created successfully")
    if excel_success:
        logger.info("✅ Excel file created successfully")
    
    # Show sample of data
    print("\nSample of scraped data:")
    if not df.empty:
        print(df[["app_name", "vendor", "price", "purchases", "rating"]].head())
    else:
        print("No data scraped")
    
    return df

if __name__ == "__main__":
    main()