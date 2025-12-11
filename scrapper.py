import os
import re
import sys
import json
import torch
import requests
import logging
from time import sleep
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scraper.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Configuration
# Configuration
# You can override this via command line argument: python scrapper.py "https://yoursite.com"
DEFAULT_START_URL = "https://www.diamondvalleyhonda.com/new-inventory/index.htm"
START_URL = DEFAULT_START_URL
MAX_PAGES = 50
MAX_CATEGORIES = 20
CHUNK_SIZE = 800
REQUEST_TIMEOUT = 15
REQUEST_RETRIES = 3
REQUEST_BACKOFF = 1.5
SELENIUM_TIMEOUT = 120
USER_AGENT = "Mozilla/5.0"

# Global variables for lazy loading
_tokenizer = None
_model = None
_device = None

def get_model():
    """Lazy load the model and tokenizer only when needed."""
    global _tokenizer, _model, _device
    if _model is None:
        logger.info("[Model] Loading pretrained FLAN-T5 model (Lite version)...")
        # Switched to flan-t5-large for speed. Use 'xl' only if quality is poor.
        model_name = "google/flan-t5-large"
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Use device_map="auto" if accelerate is installed, otherwise manual
            if torch.cuda.is_available():
                _device = "cuda"
            else:
                _device = "cpu"
                
            _model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            _model.to(_device)
            _model.eval()
            
            logger.info("[Model] Ready on %s, vocab size = %s", _device, len(_tokenizer))
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    return _tokenizer, _model, _device


def fetch_listing_html(url, timeout=REQUEST_TIMEOUT):
    """Try loading a URL via requests with basic retries/backoff."""
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.warning(
                "Failed to load listing via requests (attempt %s/%s): %s",
                attempt,
                REQUEST_RETRIES,
                e,
            )
            if attempt < REQUEST_RETRIES:
                sleep(REQUEST_BACKOFF * attempt)
    return None


def fetch_html_with_selenium(url, timeout=SELENIUM_TIMEOUT):
    """Load URL using Selenium (headless Chrome)."""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.page_load_strategy = "eager"

    logger.info("[Browser] Loading: %s", url)
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.set_page_load_timeout(timeout)
        driver.set_script_timeout(timeout)
        driver.get(url)

        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
        )

        html = driver.page_source
        logger.info("[Browser] ✅ Page loaded successfully (%s chars)", len(html))
    except Exception as e:
        logger.error("❌ Failed to load %s: %s", url, e)
        html = None
    finally:
        driver.quit()

    return html


def extract_vehicle_links(html, base_url):
    """
    Extracts only actual vehicle detail links from the page.
    Filters out generic pages like /inventory/index.htm or /specials/.
    """
    print(f"[Link Extraction] Extracting vehicle detail links...")
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        if not href or href.startswith("#") or "mailto:" in href or "tel:" in href:
            continue

        full_url = urljoin(base_url, href)

        # Generic patterns for vehicle detail pages found on most dealer sites
        vehicle_patterns = [
            r"/(?:new|used|certified|inventory)/[^/]+/[0-9]{4}-", # Common: /new/Ford/2024-...
            r"/(?:new|used|certified|inventory)/[^/]+/id", # Common: /inventory/make/model/id/...
            r"/(?:vehicle-details|detail)/", # Common: /vehicle-details/...
            r"/[0-9]{4}-[a-zA-Z0-9\-]+-[0-9a-fA-F]+", # Year-Make-Model-Hash
            r"/[0-9]{4}-[a-zA-Z0-9\-]+\.htm", # Year-Make-Model.htm
            r"view/details/",
            r"type=(?:new|used)",
        ]

        # Exclude common non-vehicle pages
        exclude_patterns = [
            r"/search", r"/compare", r"/promotion", r"/specials", r"/service", r"/parts", 
            r"/financing", r"/about", r"/contact", r"/login", r"print", r"email"
        ]

        is_excluded = any(re.search(p, href, re.IGNORECASE) for p in exclude_patterns)
        if is_excluded:
            continue

        is_vehicle_link = False
        for pattern in vehicle_patterns:
            if re.search(pattern, href, re.IGNORECASE):
                is_vehicle_link = True
                break
        
        # Heuristic: URL contains Year and specific keywords usually implies a car listing
        if not is_vehicle_link:
            if re.search(r"20[0-2][0-9]", href) and any(x in href.lower() for x in ["new", "used", "inventory", "detail"]):
                is_vehicle_link = True

        if is_vehicle_link:
            links.add(full_url)

    print(f"[Link Extraction] Found {len(links)} vehicle detail links.")
    return list(links)


def clean_and_parse_json(response):
    """Clean the raw LLM response and attempt to parse it as JSON."""
    cleaned = response.strip()

    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)

    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        cleaned = cleaned[json_start : json_end + 1]

    if not cleaned.startswith("{"):
        cleaned = "{ " + cleaned
    if not cleaned.endswith("}"):
        cleaned = cleaned + " }"

    # Fix common JSON syntax errors
    cleaned = cleaned.replace("$", "")
    # unify quotes (dumb replacement but effective for simple cases)
    cleaned = cleaned.replace(""", '"').replace(""", '"').replace("'", '"') 
    
    # Try to quote unquoted keys
    cleaned = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', cleaned)
    
    # Fix trailing commas
    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*]", "]", cleaned)
    
    # Fix objects stuck together
    cleaned = re.sub(r"}\s*{", "},{", cleaned)

    # Boolean/Null fixes
    cleaned = re.sub(r":\s*true\s*([,}])", r": true\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r":\s*false\s*([,}])", r": false\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r":\s*null\s*([,}])", r": null\1", cleaned, flags=re.IGNORECASE)

    try:
        data = json.loads(cleaned)
        return data
    except json.JSONDecodeError:
        # Last ditch effort: find anything that looks like JSON
        try:
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass
        return None


def get_vehicle_text_chunks_from_html(html, chunk_size=2000):
    """Extract relevant text from HTML and split into chunks."""
    soup = BeautifulSoup(html, "html.parser")
    sections = []

    main = soup.find("main")
    if main:
        sections.append(main.get_text("\n", strip=True))

    section = soup.find("section")
    if section and section != main:
        sections.append(section.get_text("\n", strip=True))

    for div in soup.find_all("div", class_=re.compile(r"(vehicle|detail)", re.I)):
        txt = div.get_text("\n", strip=True)
        if txt:
            sections.append(txt)

    for div in soup.find_all("div", id=re.compile(r"(vehicle|detail)", re.I)):
        txt = div.get_text("\n", strip=True)
        if txt:
            sections.append(txt)

    text = "\n".join(sections)
    if len(text.strip()) < 100:
        body = soup.body
        if body:
            text = body.get_text("\n", strip=True)
    if len(text.strip()) < 100:
        text = soup.get_text("\n", strip=True)

    lines = text.split("\n")
    chunks, buf = [], ""
    for line in lines:
        if len(buf) + len(line) + 1 <= chunk_size:
            buf += line + "\n"
        else:
            chunks.append(buf)
            buf = line + "\n"
    if buf:
        chunks.append(buf)
    return chunks


def extract_vehicle_data_from_raw_text(raw_text, vehicle_id, vehicle_url):
    """
    This function receives raw vehicle listing text and processes it to extract the relevant fields.
    It returns the structured data in the required JSON format using generic regex extraction.
    """
    make = "Unknown"
    model = "Unknown"
    year = 0
    price = 0.0
    mileage = 0
    color = "Unknown"
    vin = ""
    stock_number = ""
    trim = "Unknown"
    body_style = ""
    doors = "Unknown"
    engine = "Unknown"
    engine_size = "Unknown"
    horsepower = "Unknown"
    torque = "Unknown"
    transmission = "Unknown"
    drivetrain = "Unknown"
    fuel_type = "Unknown"
    fuel_capacity = "Unknown"
    mpg_city = "Unknown"
    mpg_highway = "Unknown"
    seating_capacity = "Unknown"
    cargo_capacity = "Unknown"
    wheelbase = "Unknown"
    length = "Unknown"
    width = "Unknown"
    height = "Unknown"
    curb_weight = "Unknown"
    ground_clearance = "Unknown"

    safety_features = []
    tech_features = []
    interior_features = []
    exterior_features = []
    performance_features = []
    standard_equipment = []
    optional_equipment = []

    # Generic Make/Model detection looking for labels
    # e.g. "Make: Toyota", "Model: Camry"
    make_match = re.search(r"(?:Make|Manufacturer):\s*([A-Za-z0-9\-]+)", raw_text, re.IGNORECASE)
    if make_match:
        make = make_match.group(1).strip()
    
    model_match = re.search(r"Model:\s*([A-Za-z0-9\-\s]+?)(?:\n|$)", raw_text, re.IGNORECASE)
    if model_match:
        model = model_match.group(1).strip()
    
    # Try textual patterns (e.g. "2024 Honda Civic")
    # This acts as a backup if explicit labels aren't found
    if make == "Unknown" or model == "Unknown":
        header_match = re.search(r"(20[0-2][0-9])\s+([A-Z][a-z]+)\s+([A-Z0-9][a-zA-Z0-9\-\s]+)", raw_text)
        if header_match:
            if year == 0: year = int(header_match.group(1))
            if make == "Unknown": make = header_match.group(2)
            # Rough guess for model, might include trim
            if model == "Unknown": model = header_match.group(3).split("\n")[0].strip()

    year_match = re.search(r"20(2[0-9])|20(1[0-9])|20(0[0-9])", raw_text)
    if year_match and year == 0:
        # Construct the full year from the groups
        y_str = year_match.group(0)
        year = int(y_str)

    # Price: look for larger numbers with $ sign, picking the first reasonable one
    price_matches = re.finditer(r"\$([0-9]{1,3}(?:,[0-9]{3})*)", raw_text)
    for pm in price_matches:
        p_val = float(pm.group(1).replace(",", ""))
        if p_val > 1000: # filter out small prices like accessories
            price = p_val
            break

    # Mileage
    mileage_match = re.search(r"(?:Mileage|Odometer):\s*([0-9,]+)", raw_text, re.IGNORECASE)
    if mileage_match:
        mileage = int(mileage_match.group(1).replace(",", ""))
    else:
        # Look for "XX,XXX miles" patterns
        m_match = re.search(r"([0-9,]{1,7})\+?\s*(?:miles|mi\.)", raw_text, re.IGNORECASE)
        if m_match:
            try:
                mileage = int(m_match.group(1).replace(",", ""))
            except:
                pass

    # Color
    color_match = re.search(r"(?:Exterior Color|Color):\s*([A-Za-z\s]+)", raw_text, re.IGNORECASE)
    if color_match:
        color = color_match.group(1).strip()

    # VIN
    vin_match = re.search(r"[A-HJ-NPR-Z0-9]{17}", raw_text)
    if vin_match:
        vin = vin_match.group(0)

    # Stock Number
    stock_match = re.search(r"(?:Stock Number|Stock #|Stock No\.?):\s*([A-Z0-9\-]+)", raw_text, re.IGNORECASE)
    if stock_match:
        stock_number = stock_match.group(1).strip()

    # Trim
    trim_match = re.search(r"Trim:\s*([A-Za-z0-9\-\s]+)", raw_text, re.IGNORECASE)
    if trim_match:
        trim = trim_match.group(1).strip()

    # Body Style
    body_match = re.search(r"(?:Body Style|Body):\s*([A-Za-z\s]+)", raw_text, re.IGNORECASE)
    if body_match:
        body_style = body_match.group(1).strip()

    # Transmission
    trans_match = re.search(r"(?:Transmission|Trans):\s*([A-Za-z0-9\-\s\.]+)", raw_text, re.IGNORECASE)
    if trans_match:
        transmission = trans_match.group(1).strip()

    # Engine
    eng_match = re.search(r"Engine:\s*([A-Za-z0-9\-\s\.]+)", raw_text, re.IGNORECASE)
    if eng_match:
        engine = eng_match.group(1).strip()

    # Drivetrain
    drive_match = re.search(r"(?:Drivetrain|Drive Type):\s*([A-Za-z0-9\-\s]+)", raw_text, re.IGNORECASE)
    if drive_match:
        drivetrain = drive_match.group(1).strip()

    # Note: Regex extraction for features is very brittle across sites. 
    # Valid strategy: If safety/tech words appear, add them.
    # We use a reduced generic keyword set
    common_features = {
        "safety": ["Lane Departure", "Blind Spot", "Backup Camera", "Airbag", "ABS"],
        "tech": ["Bluetooth", "Navigation", "CarPlay", "Android Auto", "USB"],
        "interior": ["Leather", "Heated Seats", "Sunroof", "Moonroof"],
    }
    
    for category, keywords in common_features.items():
        for k in keywords:
            if k.lower() in raw_text.lower():
                if category == "safety": safety_features.append(k)
                elif category == "tech": tech_features.append(k)
                elif category == "interior": interior_features.append(k)

    vehicle_data = {
        "id": vehicle_id,
        "make": make,
        "model": model,
        "year": year,
        "price": price,
        "mileage": mileage,
        "color": color,
        "vin": vin,
        "stockNumber": stock_number,
        "condition": "new" if mileage < 1000 and mileage != 0 else "used", # Simple heuristic
        "detail_url": vehicle_url,
        "options": {
            "trim": trim,
            "bodyStyle": body_style,
            "doors": doors,
            "engine": engine,
            "engineSize": engine_size,
            "horsepower": horsepower,
            "torque": torque,
            "transmission": transmission,
            "drivetrain": drivetrain,
            "fuelType": fuel_type,
            "fuelCapacity": fuel_capacity,
            "mpgCity": mpg_city,
            "mpgHighway": mpg_highway,
            "seatingCapacity": seating_capacity,
            "cargoCapacity": cargo_capacity,
            "wheelbase": wheelbase,
            "length": length,
            "width": width,
            "height": height,
            "curbWeight": curb_weight,
            "groundClearance": ground_clearance,
            "safetyFeatures": safety_features,
            "techFeatures": tech_features,
            "interiorFeatures": interior_features,
            "exteriorFeatures": exterior_features,
            "performanceFeatures": performance_features,
            "standardEquipment": standard_equipment,
            "optionalEquipment": optional_equipment,
        },
    }
    return vehicle_data


def extract_with_flan(vehicle_id, vehicle_url, text, current_data=None):
    """
    Use local FLAN-T5 model to dynamically extract structured vehicle data as JSON.
    This is a fallback/enhancement method when regex fails.
    """
    logger.info(f"🧠 Invoking FLAN-T5 for vehicle #{vehicle_id}...")
    tokenizer, model, device = get_model()

    # Prompt
    prompt = f"""
    Extract vehicle specifications from the text below and return strictly valid JSON.
    
    Keys required: make, model, year, price, mileage, color, vin, bodyStyle, engine, transmission.
    If unknown, use "Unknown" or 0.

    Text:
    {text[:1500]}
    
    JSON:
    """

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=512, 
            num_beams=3,
            early_stopping=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # logger.debug(f"Raw LLM response: {response}")

    data = clean_and_parse_json(response)
    
    if data:
        # Merge with current_data if provided
        if current_data:
            # We trust LLM to fill in gaps, but maybe Regex is more precise for numbers.
            # Let's simple fill unknowns
            for k, v in data.items():
                if k in current_data:
                    is_unknown = current_data[k] in [None, "Unknown", 0, "", "0"]
                    if is_unknown and v not in [None, "Unknown", 0, ""]:
                       current_data[k] = v
            return current_data
        
        return {
            "id": vehicle_id,
            "detail_url": vehicle_url,
            **data
        }
    
    return current_data 


def get_next_page_url(html, base_url):
    """Extract the next page URL from pagination elements"""
    soup = BeautifulSoup(html, "html.parser")

    pagination_selectors = [
        "a.next",
        "a[aria-label='Next']",
        "a[title='Next']",
        ".pagination a.next",
        ".pager a.next",
        "a:-soup-contains('Next')",
        "a:-soup-contains('>')",
        "a:-soup-contains('→')",
        ".pagination .next",
        ".pager .next",
        "a[class*='next']",
        "a[class*='page-next']",
        "a[class*='pagination-next']",
        "a[class*='btn-next']",
        "a[class*='nav-next']",
        ".pagination li:last-child a",
        ".pager li:last-child a",
        "a[rel='next']",
        "link[rel='next']",
    ]

    for selector in pagination_selectors:
        try:
            next_element = soup.select_one(selector)
            if next_element and next_element.get("href"):
                next_url = urljoin(base_url, next_element["href"])
                if next_url != base_url and next_url != base_url.rstrip("/"):
                    print(f"🔗 Found next page using selector '{selector}': {next_url}")
                    return next_url
        except Exception as e:
            continue

    page_links = soup.find_all("a", href=True)
    current_page_num = None

    if "page=" in base_url:
        try:
            current_page_num = int(base_url.split("page=")[1].split("&")[0])
        except:
            pass
    elif "p=" in base_url:
        try:
            current_page_num = int(base_url.split("p=")[1].split("&")[0])
        except:
            pass

    for link in page_links:
        href = link.get("href", "")
        if "page=" in href or "p=" in href:
            try:
                if "page=" in href:
                    page_num = int(href.split("page=")[1].split("&")[0])
                else:
                    page_num = int(href.split("p=")[1].split("&")[0])

                if current_page_num and page_num == current_page_num + 1:
                    next_url = urljoin(base_url, href)
                    print(f"🔗 Found next page using page number: {next_url}")
                    return next_url
            except:
                continue

    load_more_selectors = [
        "a:-soup-contains('Show More')",
        "a:-soup-contains('Load More')",
        "a:-soup-contains('View More')",
        "a:-soup-contains('See More')",
        "button:-soup-contains('Show More')",
        "button:-soup-contains('Load More')",
        ".load-more a",
        ".show-more a",
        ".pagination-load-more a",
        ".btn-load-more",
        ".load-more-btn",
    ]

    for selector in load_more_selectors:
        try:
            load_more_element = soup.select_one(selector)
            if load_more_element and load_more_element.get("href"):
                next_url = urljoin(base_url, load_more_element["href"])
                if next_url != base_url and next_url != base_url.rstrip("/"):
                    print(
                        f"🔗 Found next page using load more selector '{selector}': {next_url}"
                    )
                    return next_url
        except Exception as e:
            continue

    print("❌ No next page found")
    return None


def discover_inventory_categories(html, base_url):
    """Discover different inventory categories (new, used, different car types)"""
    soup = BeautifulSoup(html, "html.parser")
    categories = []

    category_selectors = [
        "a[href*='new-inventory']",
        "a[href*='used-inventory']",
        "a[href*='inventory']",
        "a[href*='new/']",
        "a[href*='used/']",
        ".inventory-nav a",
        ".category-nav a",
        ".filter-nav a",
        "a[class*='category']",
        "a[class*='inventory']",
        "a[class*='filter']",
    ]

    for selector in category_selectors:
        try:
            elements = soup.select(selector)
            for element in elements:
                href = element.get("href")
                if href:
                    full_url = urljoin(base_url, href)
                    if any(
                        keyword in full_url.lower()
                        for keyword in [
                            "inventory",
                            "new",
                            "used",
                        ]
                    ):
                        if full_url not in categories and full_url != base_url:
                            categories.append(full_url)
        except Exception as e:
            continue

    return categories


def discover_additional_inventory_urls(base_url):
    """Discover additional inventory URLs by trying common patterns"""
    additional_urls = []

    patterns = [
        f"{base_url.rstrip('/')}/used-inventory/index.htm",
        f"{base_url.rstrip('/')}/new-inventory/index.htm",
        f"{base_url.rstrip('/')}/inventory/index.htm",
        f"{base_url.rstrip('/')}/used-inventory/",
        f"{base_url.rstrip('/')}/new-inventory/",
        f"{base_url.rstrip('/')}/inventory/",
        f"{base_url.rstrip('/')}/used/",
        f"{base_url.rstrip('/')}/new/",
        f"{base_url.rstrip('/')}/all-inventory/",
    ]
    return patterns


def main_scraper(url, max_pages=MAX_PAGES, max_categories=MAX_CATEGORIES):
    """
    Enhanced main scraper with comprehensive pagination and category discovery
    """
    output_file = "vehicle_inventory.json"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("[\n")

    first = True
    total_vehicles = 0
    processed_urls = set()
    categories_to_process = [url]
    processed_categories = set()

    print(f"🚀 Starting comprehensive inventory scraping...")
    print(f"📊 Max pages per category: {max_pages}")
    print(f"📊 Max categories to process: {max_categories}")

    print(f"\n{'='*60}")
    print(f"🔍 DISCOVERING INVENTORY CATEGORIES")
    print(f"{'='*60}")

    initial_html = fetch_html_with_selenium(url)
    if initial_html:
        discovered_categories = discover_inventory_categories(initial_html, url)
        categories_to_process.extend(discovered_categories)
        print(
            f"🎯 Discovered {len(discovered_categories)} categories from page analysis"
        )

        additional_patterns = discover_additional_inventory_urls(url)
        categories_to_process.extend(additional_patterns)
        print(f"🎯 Added {len(additional_patterns)} additional URL patterns")

        total_discovered = len(discovered_categories) + len(additional_patterns)
        print(f"📊 Total categories to process: {total_discovered}")
        # Only print first few
        for i, cat in enumerate(categories_to_process[:5], 1):
            print(f"  {i}. {cat}")
        if len(categories_to_process) > 5:
            print(f"  ... and {len(categories_to_process) - 5} more")

    category_count = 0
    for category_url in categories_to_process:
        if category_count >= max_categories:
            print(f"⚠️ Reached maximum category limit ({max_categories}). Stopping.")
            break

        if category_url in processed_categories:
            continue

        category_count += 1
        processed_categories.add(category_url)

        print(f"\n{'='*60}")
        print(f"📂 PROCESSING CATEGORY {category_count}")
        print(f"🌐 Category URL: {category_url}")
        print(f"{'='*60}")

        page_number = 1
        current_page_url = category_url
        category_vehicles = 0

        while current_page_url and page_number <= max_pages:
            print(f"\n📄 Processing page {page_number}...")
            print(f"🔗 URL: {current_page_url}")

            if current_page_url in processed_urls:
                print(f"⚠️ URL already processed, skipping: {current_page_url}")
                break

            processed_urls.add(current_page_url)

            inventory_html = fetch_html_with_selenium(current_page_url)
            if not inventory_html:
                print(f"❌ Failed to load page: {current_page_url}")
                break

            # print(f"🔍 Extracting vehicle links...")
            vehicle_links = extract_vehicle_links(inventory_html, current_page_url)
            # print(f"🔗 Found {len(vehicle_links)} vehicle detail links")

            if not vehicle_links:
                print(f"⚠️ No vehicle links found, moving to next page")
                next_page_url = get_next_page_url(inventory_html, current_page_url)
                if next_page_url:
                    current_page_url = next_page_url
                    page_number += 1
                    continue
                else:
                    break

            print(
                f"\n🚗 Processing {len(vehicle_links)} vehicles from page {page_number}..."
            )

            for i, vehicle_url in enumerate(vehicle_links, start=1):
                print(
                    f"\n🚗 Processing vehicle {i}/{len(vehicle_links)}: {vehicle_url}"
                )

                page_html = fetch_html_with_selenium(vehicle_url)
                if not page_html:
                    print(f"Skipping {vehicle_url} due to load failure")
                    continue

                html_chunks = get_vehicle_text_chunks_from_html(
                    page_html, chunk_size=CHUNK_SIZE
                )
                
                # Use only first 2 chunks for efficiency (listing is usually top)
                all_raw_text = ""
                for chunk in html_chunks[:3]:
                    all_raw_text += chunk + "\n"

                # print(f"  > Extracting structured data (Regex)...")
                vehicle_data = extract_vehicle_data_from_raw_text(
                    all_raw_text, total_vehicles + 1, vehicle_url
                )

                # Quality Check
                is_low_quality = (
                    vehicle_data["price"] == 0.0 or 
                    vehicle_data["make"] == "Unknown" or 
                    vehicle_data["year"] == 0
                )
                
                if is_low_quality:
                    print("  ⚠️ Low quality extraction (missing information). Enhancing with LLM...")
                    try:
                        vehicle_data = extract_with_flan(total_vehicles + 1, vehicle_url, all_raw_text, current_data=vehicle_data)
                        print("  ✨ LLM enhancement successful.")
                    except Exception as e:
                        print(f"  ❌ LLM failed: {e}")

                if vehicle_data:
                    # Print summary
                    print(f"  ✔ Results: {vehicle_data.get('year')} {vehicle_data.get('make')} {vehicle_data.get('model')} - ${vehicle_data.get('price')}")
                    
                    with open(output_file, "a", encoding="utf-8") as f:
                        if not first:
                            f.write(",\n")
                        f.write(json.dumps(vehicle_data, ensure_ascii=False, indent=2))
                        first = False
                    total_vehicles += 1
                    category_vehicles += 1

            print(f"\n🔍 Looking for next page in category...")
            next_page_url = get_next_page_url(inventory_html, current_page_url)

            if next_page_url:
                # print(f"✅ Found next page: {next_page_url}")
                current_page_url = next_page_url
                page_number += 1
            else:
                print(f"❌ No more pages found in this category.")
                break

        print(
            f"\n📊 Category {category_count} complete: {category_vehicles} vehicles scraped"
        )

    with open(output_file, "a", encoding="utf-8") as f:
        f.write("]\n")

    print(f"\n{'='*60}")
    print(f"✅ COMPREHENSIVE SCRAPING COMPLETE!")
    print(f"📊 Total categories processed: {category_count}")
    print(f"📊 Total pages processed: {len(processed_urls)}")
    print(f"🚗 Total vehicles scraped: {total_vehicles}")
    print(f"💾 Data saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_url = sys.argv[1]
        print(f"🎯 Using provided target URL: {target_url}")
    else:
        target_url = START_URL
        print(f"⚠️ No URL provided in arguments. Using default: {target_url}")
        print(f"💡 Usage: python scrapper.py <url>")

    main_scraper(
        target_url,
        max_pages=MAX_PAGES,
        max_categories=MAX_CATEGORIES,
    )
