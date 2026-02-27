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
# Key agricultural districts by state
# (major crop-producing districts listed; less common → "Other / Not Listed")
# ---------------------------------------------------------------------------
DISTRICTS_BY_STATE: dict[str, list[str]] = {
    "Andhra Pradesh": [
        "Guntur", "Krishna", "Kurnool", "East Godavari", "West Godavari",
        "Prakasam", "Srikakulam", "Vizianagaram", "Visakhapatnam",
        "Nellore", "Chittoor", "Kadapa", "Anantapur", "Other / Not Listed",
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
    "Chhattisgarh": [
        "Raipur", "Bilaspur", "Durg", "Rajnandgaon", "Raigarh",
        "Korba", "Janjgir-Champa", "Bastar", "Surguja", "Other / Not Listed",
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
}
# Fallback for states with no explicit district list
DEFAULT_DISTRICTS = ["Other / Not Listed (state-level data will be used)"]


# ---------------------------------------------------------------------------
# Ensure directories exist (called when pipeline / app starts)
# ---------------------------------------------------------------------------
def ensure_dirs():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
