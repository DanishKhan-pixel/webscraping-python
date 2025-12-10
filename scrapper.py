import os
import re
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
START_URL = "https://www.diamondvalleyhonda.com/new-inventory/index.htm"
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

        vehicle_patterns = [
            r"/(new|used)/[A-Za-z\-]+/\d{4}-[A-Za-z0-9\-]+-[a-f0-9]+\.htm",
            r"/(new|used)/[A-Za-z\-]+/\d{4}-[A-Za-z0-9\-]+\.htm",
            r"/(new|used)/[A-Za-z\-]+/\d{4}-[A-Za-z0-9\-]+.*\.htm",
            r"/new/[A-Za-z\-]+/\d{4}-[A-Za-z0-9\-]+.*\.htm",
            r"/used/[A-Za-z\-]+/\d{4}-[A-Za-z0-9\-]+.*\.htm",
        ]

        general_vehicle_patterns = [
            r".*civic.*\.htm",
            r".*accord.*\.htm",
        ]

        is_vehicle_link = False
        for pattern in vehicle_patterns:
            if re.search(pattern, href, re.IGNORECASE):
                is_vehicle_link = True
                break

        if not is_vehicle_link:
            for pattern in general_vehicle_patterns:
                if re.search(pattern, href, re.IGNORECASE):
                    is_vehicle_link = True
                    break

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
    It returns the structured data in the required JSON format using direct regex extraction.
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

    if "Honda" in raw_text:
        make = "Honda"

    model_patterns = [
        "Civic",
        "Accord",
        "CR-V",
        "Pilot",
        "HR-V",
        "Passport",
        "Ridgeline",
        "Odyssey",
        "Insight",
    ]
    for pattern in model_patterns:
        if pattern in raw_text:
            model = pattern
            break

    year_match = re.search(r"20(2[0-9])", raw_text)
    if year_match:
        year = int("20" + year_match.group(1))

    price_match = re.search(r"\$([0-9,]+)", raw_text)
    if price_match:
        price = float(price_match.group(1).replace(",", ""))

    color_patterns = [
        "Solar Silver",
        "Crystal Black Pearl",
        "Rallye Red",
        "Platinum White Pearl",
        "Black",
        "White",
        "Silver",
        "Blue",
        "Red",
        "Gray",
        "Grey",
    ]
    for color_pattern in color_patterns:
        if color_pattern in raw_text:
            color = color_pattern
            break

    interior_color = "Unknown"
    if "Interior Color" in raw_text:
        interior_match = re.search(r"Interior Color\s*\n\s*([^\n]+)", raw_text)
        if interior_match:
            interior_color = interior_match.group(1).strip()

    vin_match = re.search(r"[A-HJ-NPR-Z0-9]{17}", raw_text)
    if vin_match:
        vin = vin_match.group(0)

    stock_match = re.search(r"Stock Number\s*([A-Z0-9]+)", raw_text)
    if stock_match:
        stock_number = stock_match.group(1)

    trim_patterns = ["LX", "EX", "Sport", "Touring", "Type R", "Si"]
    for pattern in trim_patterns:
        if pattern in raw_text:
            trim = pattern
            break

    body_style_patterns = ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible", "Wagon"]
    for pattern in body_style_patterns:
        if pattern in raw_text:
            body_style = pattern
            break

    doors_match = re.search(r"(\d+)\s*doors?", raw_text)
    if doors_match:
        doors = doors_match.group(1)
    elif "Sedan" in raw_text:
        doors = "4"
    elif "SUV" in raw_text or "Hatchback" in raw_text:
        doors = "5"

    engine_patterns = ["I-4 cyl", "V6", "V8", "2.0L", "1.5L", "3.5L", "Turbo"]
    for pattern in engine_patterns:
        if pattern in raw_text:
            engine = pattern
            break

    engine_size_match = re.search(r"Engine liters:\s*([0-9.]+L)", raw_text)
    if engine_size_match:
        engine_size = engine_size_match.group(1)

    hp_match = re.search(r"Horsepower:\s*([0-9]+)hp", raw_text)
    if hp_match:
        horsepower = hp_match.group(1) + "hp"

    torque_match = re.search(r"Torque:\s*([0-9]+)\s*lb\.-ft\.", raw_text)
    if torque_match:
        torque = torque_match.group(1) + " lb.-ft."

    transmission_patterns = [
        "CVT",
        "Automatic",
        "Manual",
        "6-speed",
        "8-speed",
        "9-speed",
    ]
    for pattern in transmission_patterns:
        if pattern in raw_text:
            transmission = pattern
            break

    drivetrain_patterns = [
        "Front-Wheel Drive",
        "All-Wheel Drive",
        "Rear-Wheel Drive",
        "4WD",
        "AWD",
        "FWD",
        "RWD",
    ]
    for pattern in drivetrain_patterns:
        if pattern in raw_text:
            drivetrain = pattern
            break

    mpg_match = re.search(r"(\d+)/(\d+)\s*MPG", raw_text)
    if mpg_match:
        mpg_city = mpg_match.group(1)
        mpg_highway = mpg_match.group(2)

    seating_match = re.search(r"(\d+)\s*seats?", raw_text)
    if seating_match:
        seating_capacity = seating_match.group(1)

    if "Gasoline" in raw_text or "Gas" in raw_text:
        fuel_type = "Gasoline"
    elif "Hybrid" in raw_text:
        fuel_type = "Hybrid"
    elif "Electric" in raw_text or "EV" in raw_text:
        fuel_type = "Electric"

    fuel_cap_match = re.search(r"Fuel tank capacity:\s*([0-9.]+)gal\.", raw_text)
    if fuel_cap_match:
        fuel_capacity = fuel_cap_match.group(1) + " gal."

    length_match = re.search(
        r'Exterior length:\s*([0-9,]+)mm\s*\(([0-9.]+)"\)', raw_text
    )
    if length_match:
        length = length_match.group(2) + '"'

    width_match = re.search(
        r'Exterior body width:\s*([0-9,]+)mm\s*\(([0-9.]+)"\)', raw_text
    )
    if width_match:
        width = width_match.group(2) + '"'

    height_match = re.search(
        r'Exterior height:\s*([0-9,]+)mm\s*\(([0-9.]+)"\)', raw_text
    )
    if height_match:
        height = height_match.group(2) + '"'

    wheelbase_match = re.search(r'Wheelbase:\s*([0-9,]+)mm\s*\(([0-9.]+)"\)', raw_text)
    if wheelbase_match:
        wheelbase = wheelbase_match.group(2) + '"'

    weight_match = re.search(r"Curb weight:\s*([0-9,]+)kg\s*\(([0-9,]+)lbs\)", raw_text)
    if weight_match:
        curb_weight = weight_match.group(2) + " lbs"

    cargo_match = re.search(
        r"Interior.*cargo volume:\s*([0-9]+)\s*L\s*\(([0-9.]+)\s*cu\.ft\.\)", raw_text
    )
    if cargo_match:
        cargo_capacity = cargo_match.group(2) + " cu.ft."

    feature_keywords = {
        "safety": [
            "Lane departure",
            "ABS",
            "Traction control",
            "Airbags",
            "Blind spot",
            "Collision",
            "Safety",
        ],
        "tech": [
            "Wireless phone connectivity",
            "Exterior parking camera",
            "Navigation",
            "Bluetooth",
            "USB",
            "Touchscreen",
        ],
        "interior": [
            "Automatic temperature control",
            "Steering wheel mounted audio controls",
            "Leather",
            "Heated seats",
        ],
        "exterior": [
            "Auto high-beam headlights",
            "Fully automatic headlights",
            "LED",
            "Fog lights",
        ],
        "performance": [
            "Speed-sensing steering",
            "Four wheel independent suspension",
            "Sport mode",
        ],
        "standard": [
            "4 Speakers",
            "AM/FM radio",
            "Air Conditioning",
            "Power steering",
            "Power windows",
        ],
    }

    for category, keywords in feature_keywords.items():
        for keyword in keywords:
            if keyword in raw_text:
                if category == "safety":
                    safety_features.append(keyword)
                elif category == "tech":
                    tech_features.append(keyword)
                elif category == "interior":
                    interior_features.append(keyword)
                elif category == "exterior":
                    exterior_features.append(keyword)
                elif category == "performance":
                    performance_features.append(keyword)
                elif category == "standard":
                    standard_equipment.append(keyword)

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
        "condition": "new",
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
        "a[href*='honda-']",
        "a[href*='civic']",
        "a[href*='accord']",
        "a[href*='cr-v']",
        "a[href*='pilot']",
        "a[href*='odyssey']",
        "a[href*='passport']",
        "a[href*='ridgeline']",
        "a[href*='hr-v']",
        "a[href*='insight']",
        "a[href*='fit']",
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
                            "new/",
                            "used/",
                            "honda-",
                            "civic",
                            "accord",
                            "cr-v",
                            "pilot",
                            "odyssey",
                            "passport",
                            "ridgeline",
                            "hr-v",
                            "insight",
                            "fit",
                        ]
                    ):
                        if full_url not in categories and full_url != base_url:
                            categories.append(full_url)
        except Exception as e:
            continue

    nav_selectors = [
        "nav a",
        ".main-nav a",
        ".primary-nav a",
        ".menu a",
        ".navigation a",
        "ul.menu a",
        ".header-nav a",
    ]

    for selector in nav_selectors:
        try:
            elements = soup.select(selector)
            for element in elements:
                href = element.get("href")
                if href:
                    full_url = urljoin(base_url, href)
                    if any(
                        keyword in full_url.lower()
                        for keyword in ["inventory", "new", "used", "vehicles", "cars"]
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
    ]

    honda_models = [
        "civic",
        "accord",
        "cr-v",
        "pilot",
        "odyssey",
        "passport",
        "ridgeline",
        "hr-v",
        "insight",
        "fit",
    ]
    for model in honda_models:
        patterns.extend(
            [
                f"{base_url.rstrip('/')}/new-inventory/{model}-hemet-ca.htm",
                f"{base_url.rstrip('/')}/used-inventory/{model}-hemet-ca.htm",
                f"{base_url.rstrip('/')}/new-honda-{model}-for-sale-in-hemet-ca.htm",
                f"{base_url.rstrip('/')}/used-honda-{model}-for-sale-in-hemet-ca.htm",
            ]
        )

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
    main_scraper(
        START_URL,
        max_pages=MAX_PAGES,
        max_categories=MAX_CATEGORIES,
    )
