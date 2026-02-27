"""
Risk engine: composite risk index, disease probability, and prevention measures.

Risk components:
  1. Climate risk  — sourced from climate_vulnerability.csv (0-100 index).
                     Falls back to 50 (moderate) if dataset unavailable.
  2. Disease risk  — curated knowledge base (crop × common diseases).
                     Includes probability estimate, severity, and season.

Composite risk score (0-100):
    composite = 0.5 × climate_risk + 0.5 × disease_severity_score

Risk label thresholds:
    0-25  → Low
    26-50 → Moderate
    51-75 → High
    76+   → Very High

Disease data note:
    No single public dataset cleanly maps crop × region × disease probability.
    The DISEASE_RISK_DB below is compiled from ICAR, NIPHM, and state agriculture
    department publications and should be treated as indicative, not clinical.
    When district climate vulnerability data is available, it scales the base
    probability proportionally.
"""

from src.config import W_RISK
from src.region_data_loader import get_climate_vulnerability


# ---------------------------------------------------------------------------
# Disease knowledge base
# Each entry: { "name", "probability" (0-1 base), "severity", "season", "prevention" }
# ---------------------------------------------------------------------------
DISEASE_RISK_DB: dict[str, list[dict]] = {
    "rice": [
        {
            "name": "Rice Blast (Magnaporthe oryzae)",
            "probability": 0.45, "severity": "high", "season": "Kharif",
            "prevention": [
                "Use blast-resistant varieties (IR64, Swarna sub1).",
                "Spray Tricyclazole or Propiconazole at tillering stage.",
                "Avoid excess nitrogen application.",
                "Ensure proper field drainage.",
            ],
        },
        {
            "name": "Brown Plant Hopper (BPH)",
            "probability": 0.40, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Use BPH-resistant varieties.",
                "Spray Imidacloprid or Buprofezin.",
                "Avoid close planting; maintain proper spacing.",
            ],
        },
        {
            "name": "Bacterial Leaf Blight (Xanthomonas oryzae)",
            "probability": 0.35, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Seed treatment with Streptocycline.",
                "Avoid flood irrigation after flowering.",
                "Use copper-based bactericides.",
            ],
        },
    ],
    "maize": [
        {
            "name": "Fall Armyworm (Spodoptera frugiperda)",
            "probability": 0.55, "severity": "high", "season": "Kharif",
            "prevention": [
                "Apply Emamectin benzoate or Spinetoram at whorl stage.",
                "Mix fine sand + lime in leaf whorls (physical deterrent).",
                "Use pheromone traps for monitoring.",
            ],
        },
        {
            "name": "Maize Stem Borer (Chilo partellus)",
            "probability": 0.40, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Release Trichogramma egg parasitoids.",
                "Apply Carbofuran granules in whorls.",
                "Remove and destroy infested plants early.",
            ],
        },
        {
            "name": "Common Rust (Puccinia sorghi)",
            "probability": 0.30, "severity": "low", "season": "Rabi",
            "prevention": [
                "Spray Mancozeb or Propiconazole at early rust signs.",
                "Use rust-resistant hybrids.",
            ],
        },
    ],
    "chickpea": [
        {
            "name": "Fusarium Wilt (Fusarium oxysporum)",
            "probability": 0.40, "severity": "high", "season": "Rabi",
            "prevention": [
                "Seed treatment with Trichoderma viride.",
                "Use wilt-resistant varieties (JG-62, Annigeri).",
                "Soil solarisation before sowing.",
                "Crop rotation with non-legume crops.",
            ],
        },
        {
            "name": "Gram Pod Borer (Helicoverpa armigera)",
            "probability": 0.50, "severity": "high", "season": "Rabi",
            "prevention": [
                "Spray Indoxacarb or Chlorantraniliprole at pod formation.",
                "Install pheromone traps.",
                "Intercrop with coriander or mustard as repellent.",
            ],
        },
    ],
    "kidneybeans": [
        {
            "name": "Bean Common Mosaic Virus (BCMV)",
            "probability": 0.35, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Use virus-free certified seed.",
                "Control aphid vectors with Imidacloprid.",
                "Remove infected plants immediately.",
            ],
        },
        {
            "name": "Rust (Uromyces phaseoli)",
            "probability": 0.30, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Spray Mancozeb 75 WP at 10-day intervals.",
                "Ensure good air circulation through proper spacing.",
            ],
        },
    ],
    "pigeonpeas": [
        {
            "name": "Fusarium Wilt (Fusarium udum)",
            "probability": 0.45, "severity": "high", "season": "Kharif",
            "prevention": [
                "Seed treatment with Carbendazim.",
                "Use wilt-tolerant varieties (ICPH 2671).",
                "Long crop rotation (3-4 years) away from pigeonpea.",
            ],
        },
        {
            "name": "Pod Fly (Melanagromyza obtusa)",
            "probability": 0.40, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Spray Dimethoate or Acephate at pod-fill stage.",
                "Install yellow sticky traps.",
            ],
        },
    ],
    "mothbeans": [
        {
            "name": "Yellow Mosaic Virus",
            "probability": 0.35, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Control whitefly vectors with Imidacloprid.",
                "Use resistant varieties where available.",
                "Rogue out and destroy infected plants.",
            ],
        },
        {
            "name": "Dry Root Rot",
            "probability": 0.25, "severity": "low", "season": "Kharif",
            "prevention": [
                "Avoid water-logging.",
                "Seed treatment with Thiram + Carbendazim.",
            ],
        },
    ],
    "mungbean": [
        {
            "name": "Mungbean Yellow Mosaic Virus (MYMV)",
            "probability": 0.50, "severity": "high", "season": "Kharif",
            "prevention": [
                "Use MYMV-resistant varieties (Pusa Vishal, SML-668).",
                "Spray Imidacloprid or Thiamethoxam to control whitefly.",
                "Avoid late sowing.",
            ],
        },
        {
            "name": "Cercospora Leaf Spot",
            "probability": 0.30, "severity": "low", "season": "Kharif",
            "prevention": [
                "Spray Mancozeb or Copper oxychloride.",
                "Avoid overhead irrigation.",
            ],
        },
    ],
    "blackgram": [
        {
            "name": "Yellow Mosaic Virus (YMV)",
            "probability": 0.50, "severity": "high", "season": "Kharif",
            "prevention": [
                "Use YMV-tolerant varieties (Pant U-30, Azad Urd-1).",
                "Control whitefly early with systemic insecticides.",
                "Remove and destroy yellow-infected plants.",
            ],
        },
        {
            "name": "Fusarium Wilt",
            "probability": 0.30, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Seed treatment with Trichoderma.",
                "Soil application of Neem cake.",
            ],
        },
    ],
    "lentil": [
        {
            "name": "Stemphylium Blight (Stemphylium botryosum)",
            "probability": 0.40, "severity": "medium", "season": "Rabi",
            "prevention": [
                "Spray Iprodione or Mancozeb at first sign.",
                "Use tolerant varieties (LL-699).",
                "Avoid dense planting.",
            ],
        },
        {
            "name": "Rust (Uromyces fabae)",
            "probability": 0.30, "severity": "medium", "season": "Rabi",
            "prevention": [
                "Spray Propiconazole or Mancozeb.",
                "Early sowing helps avoid peak rust season.",
            ],
        },
    ],
    "pomegranate": [
        {
            "name": "Bacterial Blight (Xanthomonas axonopodis)",
            "probability": 0.45, "severity": "high", "season": "Year-round",
            "prevention": [
                "Spray Copper oxychloride + Streptocycline.",
                "Prune infected branches; destroy debris.",
                "Avoid overhead irrigation.",
            ],
        },
        {
            "name": "Alternaria Fruit Spot",
            "probability": 0.35, "severity": "medium", "season": "Monsoon",
            "prevention": [
                "Spray Mancozeb or Difenconazole.",
                "Bag fruits before colour development.",
            ],
        },
    ],
    "banana": [
        {
            "name": "Panama Wilt / Fusarium Wilt (Foc TR4)",
            "probability": 0.40, "severity": "high", "season": "Year-round",
            "prevention": [
                "Plant Foc-resistant varieties (Grand Naine, Robusta).",
                "Soil drench with Carbendazim + Trichoderma.",
                "Strictly avoid moving infected soil.",
            ],
        },
        {
            "name": "Sigatoka Leaf Spot (Mycosphaerella musicola)",
            "probability": 0.45, "severity": "medium", "season": "Monsoon",
            "prevention": [
                "Spray Propiconazole or Mancozeb fortnightly.",
                "Remove badly infected leaves.",
                "Ensure good drainage around plant base.",
            ],
        },
    ],
    "mango": [
        {
            "name": "Anthracnose (Colletotrichum gloeosporioides)",
            "probability": 0.50, "severity": "high", "season": "Flowering/Fruiting",
            "prevention": [
                "Spray Carbendazim or Copper fungicide at flowering.",
                "Bag fruits before maturity.",
                "Collect and destroy fallen infected fruits.",
            ],
        },
        {
            "name": "Mango Hoppers (Amritodus atkinsoni)",
            "probability": 0.55, "severity": "medium", "season": "Flowering",
            "prevention": [
                "Spray Imidacloprid or Carbaryl at panicle emergence.",
                "Avoid thick planting; maintain canopy openness.",
            ],
        },
        {
            "name": "Powdery Mildew (Oidium mangiferae)",
            "probability": 0.40, "severity": "medium", "season": "Flowering",
            "prevention": [
                "Spray Sulphur dust or Triadimefon at bud-break.",
                "Prune to improve air circulation.",
            ],
        },
    ],
    "grapes": [
        {
            "name": "Downy Mildew (Plasmopara viticola)",
            "probability": 0.55, "severity": "high", "season": "Monsoon",
            "prevention": [
                "Spray Copper fungicide + Fosetyl-Al at 10-day intervals.",
                "Install rain-shelter or canopy management.",
                "Remove infected leaves and shoots promptly.",
            ],
        },
        {
            "name": "Powdery Mildew (Uncinula necator)",
            "probability": 0.45, "severity": "medium", "season": "Dry season",
            "prevention": [
                "Dust Sulphur powder on clusters.",
                "Spray Carbendazim or Triadimefon.",
            ],
        },
    ],
    "watermelon": [
        {
            "name": "Mosaic Virus (WMV)",
            "probability": 0.40, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Control aphid vectors with mineral oil spray.",
                "Use virus-free transplants.",
                "Remove and destroy infected vines.",
            ],
        },
        {
            "name": "Fusarium Wilt (Fusarium oxysporum f.sp. niveum)",
            "probability": 0.35, "severity": "medium", "season": "Summer",
            "prevention": [
                "Use grafted plants on wilt-resistant rootstock.",
                "Soil treatment with Carbendazim.",
                "Avoid continuous cropping on same land.",
            ],
        },
    ],
    "muskmelon": [
        {
            "name": "Powdery Mildew (Podosphaera xanthii)",
            "probability": 0.45, "severity": "medium", "season": "Summer",
            "prevention": [
                "Spray Sulphur or Carbendazim at first sign.",
                "Maintain plant spacing for airflow.",
            ],
        },
        {
            "name": "Downy Mildew",
            "probability": 0.35, "severity": "medium", "season": "Monsoon",
            "prevention": [
                "Spray Metalaxyl + Mancozeb.",
                "Avoid waterlogging in field.",
            ],
        },
    ],
    "apple": [
        {
            "name": "Apple Scab (Venturia inaequalis)",
            "probability": 0.50, "severity": "high", "season": "Spring",
            "prevention": [
                "Apply protective fungicide (Captan, Mancozeb) from bud-break.",
                "Rake and destroy fallen leaves.",
                "Use scab-resistant varieties (Gala, Fuji).",
            ],
        },
        {
            "name": "Fire Blight (Erwinia amylovora)",
            "probability": 0.35, "severity": "high", "season": "Flowering",
            "prevention": [
                "Spray Streptomycin or Copper bactericide at bloom.",
                "Prune blighted wood 30 cm below visible infection.",
                "Disinfect pruning tools between cuts.",
            ],
        },
    ],
    "orange": [
        {
            "name": "Citrus Canker (Xanthomonas citri)",
            "probability": 0.45, "severity": "high", "season": "Monsoon",
            "prevention": [
                "Spray Copper oxychloride every 3 weeks during wet season.",
                "Remove and burn infected leaves and branches.",
                "Use disease-free nursery material.",
            ],
        },
        {
            "name": "Citrus Greening (HLB – Huanglongbing)",
            "probability": 0.30, "severity": "high", "season": "Year-round",
            "prevention": [
                "Control psyllid vector with Imidacloprid or Thiamethoxam.",
                "Remove and destroy infected trees promptly.",
                "Plant certified HLB-free budwood.",
            ],
        },
    ],
    "papaya": [
        {
            "name": "Papaya Ring Spot Virus (PRSV)",
            "probability": 0.55, "severity": "high", "season": "Year-round",
            "prevention": [
                "Plant PRSV-tolerant varieties (Red Lady, Pusa Nanha).",
                "Use reflective mulch to repel aphid vectors.",
                "Remove infected plants within 24 hours.",
            ],
        },
        {
            "name": "Anthracnose (Colletotrichum gloeosporioides)",
            "probability": 0.35, "severity": "medium", "season": "Monsoon",
            "prevention": [
                "Spray Mancozeb or Copper fungicide fortnightly.",
                "Harvest at correct maturity stage; avoid mechanical injury.",
            ],
        },
    ],
    "coconut": [
        {
            "name": "Bud Rot (Phytophthora palmivora)",
            "probability": 0.35, "severity": "high", "season": "Monsoon",
            "prevention": [
                "Pour Bordeaux mixture (1%) into crown at onset of monsoon.",
                "Remove and burn infected parts.",
                "Ensure good drainage around palm base.",
            ],
        },
        {
            "name": "Root Wilt (Phytoplasma)",
            "probability": 0.25, "severity": "high", "season": "Year-round",
            "prevention": [
                "Apply Tetracycline injections (CPCRI protocol).",
                "Maintain soil nutrition with balanced NPK.",
                "Use resistant tall varieties in affected zones.",
            ],
        },
    ],
    "cotton": [
        {
            "name": "Pink Bollworm (Pectinophora gossypiella)",
            "probability": 0.50, "severity": "high", "season": "Kharif",
            "prevention": [
                "Use Bt-cotton hybrids with Cry1Ac/Cry2Ab genes.",
                "Pheromone traps for monitoring.",
                "Spray Emamectin benzoate or Spinosad at peak infestation.",
            ],
        },
        {
            "name": "Alternaria Leaf Spot (Alternaria macrospora)",
            "probability": 0.35, "severity": "medium", "season": "Monsoon",
            "prevention": [
                "Spray Mancozeb or Iprodione.",
                "Avoid wet conditions; improve drainage.",
            ],
        },
        {
            "name": "Fusarium Wilt",
            "probability": 0.30, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Use wilt-resistant varieties (MCU-5, DCH-32).",
                "Soil application of Trichoderma viride.",
            ],
        },
    ],
    "jute": [
        {
            "name": "Stem Rot (Macrophomina phaseolina)",
            "probability": 0.40, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Seed treatment with Thiram.",
                "Avoid water stagnation.",
                "Spray Carbendazim at first symptoms.",
            ],
        },
        {
            "name": "Anthracnose (Colletotrichum corchori)",
            "probability": 0.30, "severity": "low", "season": "Monsoon",
            "prevention": [
                "Spray Copper fungicide at seedling stage.",
                "Use healthy certified seed.",
            ],
        },
    ],
    "coffee": [
        {
            "name": "Coffee Leaf Rust (Hemileia vastatrix)",
            "probability": 0.55, "severity": "high", "season": "Post-monsoon",
            "prevention": [
                "Spray Copper fungicide + Propiconazole before and after monsoon.",
                "Shade management to reduce humidity.",
                "Use rust-resistant varieties (Cauvery, S795).",
            ],
        },
        {
            "name": "Coffee Berry Borer (Hypothenemus hampei)",
            "probability": 0.45, "severity": "high", "season": "Year-round",
            "prevention": [
                "Install traps with ethanol+methanol lure.",
                "Spray Endosulfan or Chlorpyriphos at cherry-boring stage.",
                "Strip-pick and destroy infested berries.",
            ],
        },
    ],
}

# Severity → numeric score for composite calculation
_SEVERITY_SCORE = {"low": 20, "medium": 50, "high": 80}

# Risk label thresholds
_RISK_LABELS = [(0, 25, "Low"), (26, 50, "Moderate"), (51, 75, "High"), (76, 100, "Very High")]


def get_disease_risks(crop: str) -> list[dict]:
    """
    Return list of disease risk entries for a crop.
    Each dict: { name, probability, severity, season, prevention }.
    """
    crop_key = crop.strip().lower()
    return DISEASE_RISK_DB.get(crop_key, [])


def _disease_severity_score(disease_list: list[dict]) -> float:
    """Average severity score weighted by probability (0-100)."""
    if not disease_list:
        return 30.0   # default moderate-low if no data
    scores = [
        _SEVERITY_SCORE.get(d["severity"], 50) * d["probability"]
        for d in disease_list
    ]
    return round(sum(scores) / len(scores), 1)


def compute_composite_risk(
    climate_vulnerability: float,
    disease_list: list[dict],
    climate_weight: float = 0.5,
) -> float:
    """
    Compute composite risk index (0-100).

    Parameters
    ----------
    climate_vulnerability : float
        Climate vulnerability index for the region (0-100).
    disease_list : list[dict]
        Disease entries from get_disease_risks().
    climate_weight : float
        Proportion of composite score from climate risk (rest from disease).

    Returns
    -------
    float, composite risk score 0-100.
    """
    disease_weight = 1.0 - climate_weight
    disease_score  = _disease_severity_score(disease_list)
    composite      = climate_weight * climate_vulnerability + disease_weight * disease_score
    return round(min(100.0, max(0.0, composite)), 1)


def get_risk_label(score: float) -> str:
    """Convert numeric risk score to human-readable label."""
    for lo, hi, label in _RISK_LABELS:
        if lo <= score <= hi:
            return label
    return "Very High"


def normalise_risk_scores(crop_risks: list[dict]) -> list[dict]:
    """
    Add 'risk_score_norm' (0-1, lower = safer) for balanced scoring.
    Modifies dicts in-place.
    """
    scores = [c["risk_score"] for c in crop_risks]
    min_r, max_r = min(scores), max(scores)
    span = max_r - min_r if max_r != min_r else 1.0
    for c in crop_risks:
        c["risk_score_norm"] = round((c["risk_score"] - min_r) / span, 4)
    return crop_risks


def get_all_prevention_measures(disease_list: list[dict]) -> list[str]:
    """
    Flatten and deduplicate all prevention measures from a disease list.
    Returns a clean list of unique prevention strings.
    """
    seen = set()
    measures = []
    for d in disease_list:
        for measure in d.get("prevention", []):
            if measure not in seen:
                seen.add(measure)
                measures.append(measure)
    return measures
