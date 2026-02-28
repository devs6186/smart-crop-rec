"""
Configuration and constants for the Smart Crop Recommendation System.
Centralizes paths, column names, random seed, regional data, and scoring weights.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Base paths (project root = parent of 'src')
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Data file names — ML pipeline (existing)
# ---------------------------------------------------------------------------
RAW_DATA_FNAME   = "Crop_Recommendation.csv"
SAMPLE_DATA_FNAME = "Crop_Recommendation_sample.csv"

# ---------------------------------------------------------------------------
# Data file names — region / economic datasets (place CSVs in data/raw/)
# If files are absent the system falls back to embedded national averages.
#
# Expected schemas:
#   state_wise_yield.csv      : State_Name, District_Name, Crop, Season,
#                               Crop_Year, Area (ha), Production (tonnes)
#   market_prices.csv         : State, District, Commodity, Modal_Price (Rs/q)
#   cost_of_cultivation.csv   : Crop, State, Cost_Per_Acre (Rs)
#   climate_vulnerability.csv : State, District, Vulnerability_Index (0-100)
# ---------------------------------------------------------------------------
REGION_YIELD_FNAME       = "state_wise_yield.csv"
CROP_YIELD_FNAME         = "crop_yield.csv"   # optional: converted to training rows (Crop, Annual_Rainfall, State -> N,P,K,...)
MARKET_PRICE_FNAME       = "market_prices.csv"
COST_CULTIVATION_FNAME   = "cost_of_cultivation.csv"
CLIMATE_RISK_FNAME       = "climate_vulnerability.csv"
UNIFIED_REGION_FNAME     = "unified_crop_region_data.csv"   # cached merged table

# ---------------------------------------------------------------------------
# Feature and target column names (must match dataset)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN   = "label"
TARGET_ALIASES  = ("label", "crop", "Crop")

# ---------------------------------------------------------------------------
# ML constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5
TOP_K_CROPS       = 5     # final crops shown to user
CANDIDATES_POOL   = 12    # top-N by ML suitability to evaluate for profit ranking
MIN_SUITABILITY_PCT = 5.0 # minimum ML confidence % to be a profit-ranking candidate

# Minimum viable land (acres) per crop — crops needing more space are filtered for small holdings
# Based on ICAR/NBSS&LUP: sugarcane/cotton need bulk; pulses/vegetables work on small plots
DEFAULT_MIN_LAND_ACRES = 0.1   # unknown crops assumed viable on small plots
CROP_MIN_LAND_ACRES: dict[str, float] = {
    # Large-scale (1–2+ acres): bulk crops, plantations
    "sugarcane": 2.0, "cotton": 1.0, "jute": 1.0, "mesta": 1.0, "tobacco": 1.0,
    # Orchards / plantations (0.5+ acres)
    "coconut": 0.5, "banana": 0.5, "mango": 0.5, "apple": 0.5, "grapes": 0.5,
    "papaya": 0.5, "pomegranate": 0.5, "watermelon": 0.5, "muskmelon": 0.5,
    "orange": 0.5, "coffee": 0.5, "arecanut": 0.5, "cashewnut": 0.5,
    "cardamom": 0.5, "black pepper": 0.5,
    # Root/tuber (0.25 acres)
    "potato": 0.25, "onion": 0.25, "sweet potato": 0.25, "tapioca": 0.25,
    "ginger": 0.25, "turmeric": 0.25, "garlic": 0.25,
    # Cereals (0.2 acres)
    "rice": 0.2, "wheat": 0.2, "maize": 0.2, "bajra": 0.2, "barley": 0.2,
    "jowar": 0.2, "ragi": 0.2, "small millets": 0.2,
    # Pulses, oilseeds, spices (0.1 acres — work on small plots)
    "chickpea": 0.1, "pigeonpeas": 0.1, "mungbean": 0.1, "blackgram": 0.1,
    "lentil": 0.1, "kidneybeans": 0.1, "mothbeans": 0.1, "cowpea": 0.1,
    "peas": 0.1, "groundnut": 0.1, "soyabean": 0.1, "mustard": 0.1,
    "sesamum": 0.1, "sunflower": 0.1, "safflower": 0.1, "niger seed": 0.1,
    "dry chillies": 0.1, "coriander": 0.1, "castor seed": 0.1,
    "linseed": 0.1, "guar seed": 0.1, "horse-gram": 0.1, "khesari": 0.1,
    "sannhamp": 0.1,
}

# ---------------------------------------------------------------------------
# Model artifact names
# ---------------------------------------------------------------------------
MODEL_ARTIFACT_NAME   = "model.joblib"
SCALER_ARTIFACT_NAME  = "scaler.joblib"
ENCODER_ARTIFACT_NAME = "label_encoder.joblib"
METADATA_FNAME        = "metadata.json"

# ---------------------------------------------------------------------------
# Profit / scoring configuration
# ---------------------------------------------------------------------------
SCORING_MODE  = "profit"   # "profit" | "balanced"
W_SUITABILITY = 0.30       # weight used only in "balanced" mode
W_PROFIT      = 0.50
W_RISK        = 0.20

# Yield adjustment: effective_yield = region_avg × (BASE + CONF × suitability)
# A 100% confident recommendation gets full regional yield;
# a low-confidence one is discounted to BASE fraction.
YIELD_BASE_FACTOR = 0.60
YIELD_CONF_FACTOR = 0.40

# ---------------------------------------------------------------------------
# Bigha → acres conversion (state-specific, gazetted references, approx.)
# ---------------------------------------------------------------------------
BIGHA_TO_ACRES: dict[str, float] = {
    "Andhra Pradesh":                           0.40,
    "Arunachal Pradesh":                        0.40,
    "Assam":                                    0.33,
    "Bihar":                                    0.20,
    "Chhattisgarh":                             0.30,
    "Goa":                                      0.40,
    "Gujarat":                                  0.40,
    "Haryana":                                  0.40,
    "Himachal Pradesh":                         0.20,
    "Jharkhand":                                0.20,
    "Karnataka":                                0.40,
    "Kerala":                                   0.40,
    "Madhya Pradesh":                           0.62,
    "Maharashtra":                              0.40,
    "Manipur":                                  0.40,
    "Meghalaya":                                0.40,
    "Mizoram":                                  0.40,
    "Nagaland":                                 0.40,
    "Odisha":                                   0.33,
    "Punjab":                                   0.40,
    "Rajasthan":                                0.62,
    "Sikkim":                                   0.20,
    "Tamil Nadu":                               0.40,
    "Telangana":                                0.40,
    "Tripura":                                  0.33,
    "Uttar Pradesh":                            0.20,
    "Uttarakhand":                              0.20,
    "West Bengal":                              0.33,
    "Delhi":                                    0.40,
    "Jammu and Kashmir":                        0.20,
    "Ladakh":                                   0.20,
    "Puducherry":                               0.40,
    "Andaman and Nicobar Islands":              0.40,
    "Chandigarh":                               0.40,
    "Dadra and Nagar Haveli and Daman and Diu": 0.40,
    "Lakshadweep":                              0.40,
}
DEFAULT_BIGHA_ACRES = 0.40   # fallback when state not in map

# ---------------------------------------------------------------------------
# Indian states list (for UI dropdown)
# ---------------------------------------------------------------------------
INDIAN_STATES: list[str] = sorted(BIGHA_TO_ACRES.keys())

# ---------------------------------------------------------------------------
# Agro-climatic zone defaults (state → N,P,K,temp,humidity,ph,rainfall)
# Used by app for region-specific inputs and by data_loader when converting crop_yield.csv.
# ---------------------------------------------------------------------------
ZONE_DEFAULTS: dict[str, dict[str, float]] = {
    "arid_nw":       {"N": 20, "P": 28, "K": 32, "temperature": 32.0, "humidity": 44.0, "ph": 7.6, "rainfall": 28.0},
    "eastern_humid": {"N": 85, "P": 42, "K": 38, "temperature": 21.0, "humidity": 90.0, "ph": 5.7, "rainfall": 295.0},
    "southern":      {"N": 72, "P": 48, "K": 45, "temperature": 24.0, "humidity": 80.0, "ph": 6.2, "rainfall": 195.0},
    "west_coast":    {"N": 84, "P": 40, "K": 36, "temperature": 20.2, "humidity": 91.0, "ph": 5.6, "rainfall": 285.0},
    "central":       {"N": 52, "P": 44, "K": 41, "temperature": 25.5, "humidity": 70.0, "ph": 6.1, "rainfall": 92.0},
    "himalayan":     {"N": 87, "P": 42, "K": 40, "temperature": 18.5, "humidity": 87.0, "ph": 5.9, "rainfall": 255.0},
    "western_dry":   {"N": 40, "P": 36, "K": 40, "temperature": 27.0, "humidity": 59.0, "ph": 6.9, "rainfall": 55.0},
}
STATE_ZONE: dict[str, str] = {
    "Rajasthan": "arid_nw", "Haryana": "arid_nw", "Punjab": "arid_nw", "Delhi": "arid_nw", "Chandigarh": "arid_nw",
    "West Bengal": "eastern_humid", "Odisha": "eastern_humid", "Assam": "eastern_humid",
    "Arunachal Pradesh": "eastern_humid", "Manipur": "eastern_humid", "Meghalaya": "eastern_humid",
    "Mizoram": "eastern_humid", "Nagaland": "eastern_humid", "Tripura": "eastern_humid",
    "Andhra Pradesh": "southern", "Telangana": "southern", "Karnataka": "southern", "Tamil Nadu": "southern", "Puducherry": "southern",
    "Kerala": "west_coast", "Goa": "west_coast",
    "Maharashtra": "western_dry", "Gujarat": "western_dry", "Dadra and Nagar Haveli and Daman and Diu": "western_dry",
    "Madhya Pradesh": "central", "Chhattisgarh": "central", "Uttar Pradesh": "central", "Bihar": "central", "Jharkhand": "central",
    "Himachal Pradesh": "himalayan", "Uttarakhand": "himalayan", "Jammu and Kashmir": "himalayan", "Ladakh": "himalayan", "Sikkim": "himalayan",
    "Andaman and Nicobar Islands": "eastern_humid", "Lakshadweep": "west_coast",
}


def get_state_soil_climate(state: str | None) -> dict[str, float]:
    """Return N,P,K,temperature,humidity,ph,rainfall for a state (from agro-climatic zone). Used for crop_yield conversion."""
    if not state or state not in STATE_ZONE:
        return {k: 50.0 if k in ("N", "P", "K") else 25.0 if k == "temperature" else 65.0 if k == "humidity" else 6.5 if k == "ph" else 120.0 for k in FEATURE_COLUMNS}
    zone = STATE_ZONE[state]
    return dict(ZONE_DEFAULTS[zone])

# ---------------------------------------------------------------------------
# Key agricultural districts by state
# (major crop-producing districts listed; less common → "Other / Not Listed")
# All states/UTs in INDIAN_STATES have an entry so district dropdown always shows options.
# ---------------------------------------------------------------------------
DISTRICTS_BY_STATE: dict[str, list[str]] = {
    "Andaman and Nicobar Islands": [
        "South Andaman", "North and Middle Andaman", "Nicobar", "Other / Not Listed",
    ],
    "Andhra Pradesh": [
        "Guntur", "Krishna", "Kurnool", "East Godavari", "West Godavari",
        "Prakasam", "Srikakulam", "Vizianagaram", "Visakhapatnam",
        "Nellore", "Chittoor", "Kadapa", "Anantapur", "Other / Not Listed",
    ],
    "Arunachal Pradesh": [
        "Papum Pare", "Changlang", "Lohit", "West Kameng", "East Siang",
        "Lower Subansiri", "Tirap", "Tawang", "Upper Siang", "Other / Not Listed",
    ],
    "Assam": [
        "Kamrup", "Nagaon", "Barpeta", "Goalpara", "Sibsagar", "Darrang",
        "Dibrugarh", "Cachar", "Tinsukia", "Jorhat", "Sonitpur", "Other / Not Listed",
    ],
    "Bihar": [
        "Patna", "Muzaffarpur", "Gaya", "Bhagalpur", "Nalanda", "Vaishali",
        "Saran", "Siwan", "East Champaran", "West Champaran", "Rohtas",
        "Aurangabad", "Darbhanga", "Samastipur", "Other / Not Listed",
    ],
    "Chandigarh": [
        "Chandigarh", "Other / Not Listed",
    ],
    "Chhattisgarh": [
        "Raipur", "Bilaspur", "Durg", "Rajnandgaon", "Raigarh",
        "Korba", "Janjgir-Champa", "Bastar", "Surguja", "Other / Not Listed",
    ],
    "Dadra and Nagar Haveli and Daman and Diu": [
        "Dadra and Nagar Haveli", "Daman", "Diu", "Other / Not Listed",
    ],
    "Delhi": [
        "North Delhi", "South Delhi", "East Delhi", "West Delhi", "Central Delhi",
        "New Delhi", "North East Delhi", "North West Delhi", "Shahdara", "Other / Not Listed",
    ],
    "Goa": [
        "North Goa", "South Goa", "Other / Not Listed",
    ],
    "Gujarat": [
        "Ahmedabad", "Anand", "Mehsana", "Kheda", "Junagadh", "Rajkot",
        "Amreli", "Surat", "Vadodara", "Bharuch", "Banaskantha",
        "Patan", "Sabarkantha", "Other / Not Listed",
    ],
    "Haryana": [
        "Karnal", "Hisar", "Sirsa", "Fatehabad", "Rohtak", "Bhiwani",
        "Jind", "Sonipat", "Ambala", "Kurukshetra", "Yamunanagar",
        "Kaithal", "Panipat", "Other / Not Listed",
    ],
    "Himachal Pradesh": [
        "Shimla", "Kangra", "Mandi", "Kullu", "Solan", "Sirmaur",
        "Una", "Hamirpur", "Bilaspur", "Chamba", "Other / Not Listed",
    ],
    "Jharkhand": [
        "Ranchi", "Dhanbad", "Hazaribagh", "Bokaro", "Giridih",
        "East Singhbhum", "West Singhbhum", "Gumla", "Other / Not Listed",
    ],
    "Karnataka": [
        "Belagavi", "Tumkur", "Mysuru", "Dharwad", "Haveri", "Davangere",
        "Shivamogga", "Mandya", "Hassan", "Raichur", "Ballari",
        "Kalaburagi", "Vijayapura", "Bengaluru Rural", "Kodagu", "Other / Not Listed",
    ],
    "Kerala": [
        "Thrissur", "Malappuram", "Palakkad", "Kozhikode", "Wayanad",
        "Kannur", "Ernakulam", "Alappuzha", "Kottayam", "Idukki",
        "Thiruvananthapuram", "Other / Not Listed",
    ],
    "Madhya Pradesh": [
        "Indore", "Bhopal", "Ujjain", "Sagar", "Hoshangabad", "Chhindwara",
        "Vidisha", "Raisen", "Rewa", "Sehore", "Dewas", "Mandsaur",
        "Neemuch", "Gwalior", "Morena", "Shivpuri", "Other / Not Listed",
    ],
    "Maharashtra": [
        "Pune", "Nashik", "Aurangabad", "Solapur", "Kolhapur", "Sangli",
        "Ahmednagar", "Satara", "Jalgaon", "Nanded", "Amravati", "Nagpur",
        "Akola", "Latur", "Osmanabad", "Other / Not Listed",
    ],
    "Odisha": [
        "Cuttack", "Puri", "Ganjam", "Kalahandi", "Bargarh", "Mayurbhanj",
        "Kendrapara", "Balasore", "Sambalpur", "Koraput", "Sundargarh",
        "Other / Not Listed",
    ],
    "Punjab": [
        "Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Bathinda",
        "Moga", "Ferozepur", "Faridkot", "Gurdaspur", "Sangrur",
        "Hoshiarpur", "Rupnagar", "Other / Not Listed",
    ],
    "Rajasthan": [
        "Jaipur", "Jodhpur", "Sikar", "Sri Ganganagar", "Hanumangarh",
        "Churu", "Bikaner", "Alwar", "Bharatpur", "Kota", "Bundi",
        "Tonk", "Nagaur", "Barmer", "Pali", "Ajmer", "Other / Not Listed",
    ],
    "Tamil Nadu": [
        "Coimbatore", "Thanjavur", "Erode", "Salem", "Tirupur", "Madurai",
        "Tirunelveli", "Villupuram", "Dharmapuri", "Vellore", "Thiruvarur",
        "Nagapattinam", "Dindigul", "Tiruchirapalli", "Other / Not Listed",
    ],
    "Telangana": [
        "Warangal", "Khammam", "Nizamabad", "Karimnagar", "Medak",
        "Nalgonda", "Mahbubnagar", "Rangareddy", "Adilabad", "Suryapet",
        "Jagtial", "Other / Not Listed",
    ],
    "Uttar Pradesh": [
        "Lucknow", "Agra", "Varanasi", "Kanpur", "Prayagraj", "Meerut",
        "Moradabad", "Muzaffarnagar", "Bareilly", "Mathura", "Aligarh",
        "Gorakhpur", "Sitapur", "Barabanki", "Hardoi", "Shahjahanpur",
        "Bahraich", "Gonda", "Lakhimpur Kheri", "Other / Not Listed",
    ],
    "Uttarakhand": [
        "Dehradun", "Haridwar", "Udham Singh Nagar", "Nainital",
        "Pauri Garhwal", "Tehri Garhwal", "Almora", "Other / Not Listed",
    ],
    "West Bengal": [
        "Murshidabad", "Bardhaman", "Nadia", "Hooghly", "North 24 Parganas",
        "South 24 Parganas", "Bankura", "Purulia", "Jalpaiguri",
        "Cooch Behar", "Malda", "Birbhum", "Other / Not Listed",
    ],
    "Jammu and Kashmir": [
        "Jammu", "Srinagar", "Anantnag", "Baramulla", "Kupwara",
        "Pulwama", "Kathua", "Udhampur", "Other / Not Listed",
    ],
    "Ladakh": [
        "Leh", "Kargil", "Other / Not Listed",
    ],
    "Lakshadweep": [
        "Kavaratti", "Amini", "Minicoy", "Other / Not Listed",
    ],
    "Manipur": [
        "Imphal East", "Imphal West", "Thoubal", "Bishnupur", "Churachandpur",
        "Senapati", "Ukhrul", "Tamenglong", "Other / Not Listed",
    ],
    "Meghalaya": [
        "East Khasi Hills", "West Khasi Hills", "Jaintia Hills", "Ri Bhoi",
        "East Garo Hills", "West Garo Hills", "South Garo Hills", "Other / Not Listed",
    ],
    "Mizoram": [
        "Aizawl", "Lunglei", "Champhai", "Mamit", "Kolasib",
        "Serchhip", "Lawngtlai", "Saiha", "Other / Not Listed",
    ],
    "Nagaland": [
        "Kohima", "Dimapur", "Mokokchung", "Tuensang", "Wokha",
        "Zunheboto", "Phek", "Mon", "Other / Not Listed",
    ],
    "Puducherry": [
        "Puducherry", "Karaikal", "Mahe", "Yanam", "Other / Not Listed",
    ],
    "Sikkim": [
        "East Sikkim", "South Sikkim", "West Sikkim", "North Sikkim", "Other / Not Listed",
    ],
    "Tripura": [
        "West Tripura", "South Tripura", "Dhalai", "North Tripura",
        "Gomati", "Khowai", "Unakoti", "Sepahijala", "Other / Not Listed",
    ],
}
# Fallback for states with no explicit district list (should not be needed if all states are above)
DEFAULT_DISTRICTS = ["Other / Not Listed (state-level data will be used)"]


# ---------------------------------------------------------------------------
# Ensure directories exist (called when pipeline / app starts)
# ---------------------------------------------------------------------------
def ensure_dirs():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
