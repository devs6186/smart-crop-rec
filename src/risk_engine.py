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
    # ── NEW CROPS (expanded from crop_params database) ──────────────────
    "wheat": [
        {
            "name": "Wheat Rust (Puccinia triticina)",
            "probability": 0.45, "severity": "high", "season": "Rabi",
            "prevention": [
                "Use rust-resistant varieties (HD-2967, PBW-343).",
                "Spray Propiconazole at first rust signs.",
                "Avoid late sowing in rust-prone zones.",
            ],
        },
        {
            "name": "Karnal Bunt (Tilletia indica)",
            "probability": 0.25, "severity": "medium", "season": "Rabi",
            "prevention": [
                "Seed treatment with Carboxin + Thiram.",
                "Use certified disease-free seed.",
                "Avoid irrigation at heading stage during humid weather.",
            ],
        },
    ],
    "potato": [
        {
            "name": "Late Blight (Phytophthora infestans)",
            "probability": 0.55, "severity": "high", "season": "Rabi",
            "prevention": [
                "Spray Mancozeb or Metalaxyl + Mancozeb at 7-day intervals.",
                "Use blight-resistant varieties (Kufri Jyoti, Kufri Bahar).",
                "Destroy volunteer plants and infected tubers.",
            ],
        },
        {
            "name": "Early Blight (Alternaria solani)",
            "probability": 0.40, "severity": "medium", "season": "Rabi",
            "prevention": [
                "Spray Mancozeb or Chlorothalonil at first symptom.",
                "Maintain adequate soil moisture.",
                "Use healthy certified seed tubers.",
            ],
        },
    ],
    "sugarcane": [
        {
            "name": "Red Rot (Colletotrichum falcatum)",
            "probability": 0.45, "severity": "high", "season": "Year-round",
            "prevention": [
                "Use disease-free setts from resistant varieties (Co 86032).",
                "Sett treatment with Carbendazim before planting.",
                "Destroy infected canes and ratoon debris.",
            ],
        },
        {
            "name": "Sugarcane Top Borer (Scirpophaga excerptalis)",
            "probability": 0.40, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Release Trichogramma parasitoids.",
                "Remove and destroy dead hearts early.",
                "Avoid excess nitrogen fertilization.",
            ],
        },
    ],
    "groundnut": [
        {
            "name": "Tikka Disease / Leaf Spot (Cercospora)",
            "probability": 0.45, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Spray Mancozeb or Chlorothalonil at 35 DAS.",
                "Use tolerant varieties (ICGV 91114).",
                "Maintain proper plant spacing.",
            ],
        },
        {
            "name": "Stem Rot (Sclerotium rolfsii)",
            "probability": 0.35, "severity": "high", "season": "Kharif",
            "prevention": [
                "Seed treatment with Trichoderma viride.",
                "Deep ploughing to bury sclerotia.",
                "Crop rotation with cereals.",
            ],
        },
    ],
    "mustard": [
        {
            "name": "Alternaria Blight (Alternaria brassicae)",
            "probability": 0.50, "severity": "high", "season": "Rabi",
            "prevention": [
                "Spray Mancozeb at flowering and pod formation.",
                "Use tolerant varieties (Pusa Bold).",
                "Destroy crop debris after harvest.",
            ],
        },
        {
            "name": "Aphid (Lipaphis erysimi)",
            "probability": 0.55, "severity": "medium", "season": "Rabi",
            "prevention": [
                "Spray Dimethoate or Imidacloprid at ETL.",
                "Early sowing (October) avoids peak aphid season.",
                "Conserve natural predators like ladybird beetles.",
            ],
        },
    ],
    "onion": [
        {
            "name": "Purple Blotch (Alternaria porri)",
            "probability": 0.45, "severity": "medium", "season": "Rabi/Kharif",
            "prevention": [
                "Spray Mancozeb + Carbendazim at 10-day intervals.",
                "Ensure proper drainage to avoid waterlogging.",
                "Use disease-free transplants.",
            ],
        },
        {
            "name": "Thrips (Thrips tabaci)",
            "probability": 0.50, "severity": "medium", "season": "Rabi",
            "prevention": [
                "Spray Fipronil or Spinosad at nymph stage.",
                "Use blue sticky traps for monitoring.",
                "Overhead irrigation helps reduce thrips population.",
            ],
        },
    ],
    "bajra": [
        {
            "name": "Downy Mildew (Sclerospora graminicola)",
            "probability": 0.45, "severity": "high", "season": "Kharif",
            "prevention": [
                "Seed treatment with Metalaxyl.",
                "Use resistant hybrids (HHB 67, ICTP 8203).",
                "Rogue out infected plants early.",
            ],
        },
        {
            "name": "Ergot (Claviceps fusiformis)",
            "probability": 0.30, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Spray Mancozeb at flowering stage.",
                "Collect and destroy honeydew-infected earheads.",
            ],
        },
    ],
    "jowar": [
        {
            "name": "Grain Mold (Fusarium/Aspergillus spp.)",
            "probability": 0.40, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Use mold-tolerant varieties (CSH-14, CSV-17).",
                "Harvest at physiological maturity; dry immediately.",
                "Avoid late sowing that exposes grain to monsoon end.",
            ],
        },
        {
            "name": "Shoot Fly (Atherigona soccata)",
            "probability": 0.50, "severity": "high", "season": "Kharif",
            "prevention": [
                "Early sowing to avoid peak fly activity.",
                "Seed treatment with Imidacloprid.",
                "Use fish-meal traps for adult fly monitoring.",
            ],
        },
    ],
    "barley": [
        {
            "name": "Stripe Rust (Puccinia striiformis)",
            "probability": 0.40, "severity": "high", "season": "Rabi",
            "prevention": [
                "Use resistant varieties (BH-393, Jyoti).",
                "Spray Propiconazole at first sign of pustules.",
                "Avoid late sowing in high-altitude areas.",
            ],
        },
        {
            "name": "Covered Smut (Ustilago hordei)",
            "probability": 0.25, "severity": "low", "season": "Rabi",
            "prevention": [
                "Seed treatment with Carboxin or Thiram.",
                "Use certified smut-free seed.",
            ],
        },
    ],
    "ragi": [
        {
            "name": "Blast (Pyricularia grisea)",
            "probability": 0.50, "severity": "high", "season": "Kharif",
            "prevention": [
                "Use blast-resistant varieties (GPU-28, MR-1).",
                "Spray Tricyclazole at neck blast stage.",
                "Avoid excessive nitrogen fertilization.",
            ],
        },
        {
            "name": "Aphid (Hysteroneura setariae)",
            "probability": 0.30, "severity": "low", "season": "Kharif",
            "prevention": [
                "Spray Dimethoate at ETL.",
                "Encourage natural predators.",
            ],
        },
    ],
    "soyabean": [
        {
            "name": "Rust (Phakopsora pachyrhizi)",
            "probability": 0.45, "severity": "high", "season": "Kharif",
            "prevention": [
                "Spray Hexaconazole or Propiconazole at R3 stage.",
                "Use tolerant varieties (JS 335, JS 9560).",
                "Early sowing avoids peak rust incidence.",
            ],
        },
        {
            "name": "Yellow Mosaic Virus (YMV)",
            "probability": 0.35, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Control whitefly vectors with Thiamethoxam.",
                "Use YMV-resistant varieties.",
                "Remove infected plants to reduce virus source.",
            ],
        },
    ],
    "sunflower": [
        {
            "name": "Alternaria Blight (Alternaria helianthi)",
            "probability": 0.45, "severity": "medium", "season": "Kharif/Rabi",
            "prevention": [
                "Spray Mancozeb at disease initiation.",
                "Use tolerant hybrids.",
                "Maintain optimum plant population.",
            ],
        },
        {
            "name": "Head Rot (Sclerotinia sclerotiorum)",
            "probability": 0.30, "severity": "high", "season": "Rabi",
            "prevention": [
                "Avoid overhead irrigation during flowering.",
                "Deep ploughing to bury sclerotia.",
                "Spray Carbendazim at head formation.",
            ],
        },
    ],
    "sesamum": [
        {
            "name": "Phytophthora Blight (Phytophthora parasitica)",
            "probability": 0.35, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Seed treatment with Metalaxyl.",
                "Ensure proper field drainage.",
                "Avoid waterlogged conditions.",
            ],
        },
        {
            "name": "Gall Fly (Asphondylia sesami)",
            "probability": 0.30, "severity": "low", "season": "Kharif",
            "prevention": [
                "Spray Dimethoate at bud stage.",
                "Collect and destroy infested buds.",
            ],
        },
    ],
    "castor seed": [
        {
            "name": "Botrytis Grey Rot (Botrytis ricini)",
            "probability": 0.40, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Spray Carbendazim or Thiophanate-methyl at spike emergence.",
                "Maintain proper plant spacing for ventilation.",
                "Avoid overhead irrigation during flowering.",
            ],
        },
        {
            "name": "Semilooper (Achaea janata)",
            "probability": 0.45, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Spray Quinalphos or Chlorpyriphos at larval stage.",
                "Hand-pick and destroy larvae when numbers are low.",
                "Use NPV (Nuclear Polyhedrosis Virus) bio-insecticide.",
            ],
        },
    ],
    "dry chillies": [
        {
            "name": "Anthracnose / Fruit Rot (Colletotrichum capsici)",
            "probability": 0.50, "severity": "high", "season": "Kharif",
            "prevention": [
                "Seed treatment with Thiram + Carbendazim.",
                "Spray Mancozeb or Copper oxychloride at 10-day intervals.",
                "Use tolerant varieties (Pusa Jwala).",
            ],
        },
        {
            "name": "Thrips (Scirtothrips dorsalis)",
            "probability": 0.55, "severity": "medium", "season": "Year-round",
            "prevention": [
                "Spray Fipronil or Spinosad.",
                "Use blue sticky traps.",
                "Intercrop with marigold as trap crop.",
            ],
        },
    ],
    "ginger": [
        {
            "name": "Soft Rot / Rhizome Rot (Pythium aphanidermatum)",
            "probability": 0.50, "severity": "high", "season": "Kharif",
            "prevention": [
                "Treat seed rhizome with Mancozeb + Carbendazim for 30 min.",
                "Ensure raised beds with good drainage.",
                "Apply Trichoderma viride to soil before planting.",
            ],
        },
        {
            "name": "Shoot Borer (Conogethes punctiferalis)",
            "probability": 0.35, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Spray Dimethoate or Malathion at first sign of boring.",
                "Prune and destroy infected tillers.",
            ],
        },
    ],
    "turmeric": [
        {
            "name": "Rhizome Rot (Pythium/Fusarium spp.)",
            "probability": 0.45, "severity": "high", "season": "Kharif",
            "prevention": [
                "Seed treatment with Mancozeb before planting.",
                "Apply Trichoderma-enriched FYM to soil.",
                "Avoid waterlogging; use raised beds.",
            ],
        },
        {
            "name": "Leaf Blotch (Taphrina maculans)",
            "probability": 0.30, "severity": "low", "season": "Kharif",
            "prevention": [
                "Spray Mancozeb or Copper oxychloride.",
                "Maintain proper drainage and spacing.",
            ],
        },
    ],
    "garlic": [
        {
            "name": "Purple Blotch (Alternaria porri)",
            "probability": 0.40, "severity": "medium", "season": "Rabi",
            "prevention": [
                "Spray Mancozeb + Carbendazim at 10-day intervals.",
                "Avoid overhead irrigation; use drip.",
                "Use disease-free planting cloves.",
            ],
        },
        {
            "name": "Stem/Bulb Nematode (Ditylenchus dipsaci)",
            "probability": 0.25, "severity": "medium", "season": "Rabi",
            "prevention": [
                "Hot water treatment of cloves (49C for 20 min) before planting.",
                "Crop rotation with non-allium crops for 3 years.",
            ],
        },
    ],
    "coriander": [
        {
            "name": "Stem Gall (Protomyces macrosporus)",
            "probability": 0.40, "severity": "medium", "season": "Rabi",
            "prevention": [
                "Use disease-free seed.",
                "Spray Mancozeb at early growth stage.",
                "Crop rotation with cereals.",
            ],
        },
        {
            "name": "Wilt (Fusarium oxysporum)",
            "probability": 0.30, "severity": "medium", "season": "Rabi",
            "prevention": [
                "Seed treatment with Trichoderma viride.",
                "Avoid waterlogged soils.",
            ],
        },
    ],
    "black pepper": [
        {
            "name": "Quick Wilt (Phytophthora capsici)",
            "probability": 0.50, "severity": "high", "season": "Monsoon",
            "prevention": [
                "Apply Bordeaux mixture (1%) at onset of monsoon.",
                "Improve drainage around vine base.",
                "Use tolerant varieties (Panniyur-1).",
            ],
        },
        {
            "name": "Pollu Disease (Colletotrichum gloeosporioides)",
            "probability": 0.35, "severity": "medium", "season": "Monsoon",
            "prevention": [
                "Spray Bordeaux mixture during spike formation.",
                "Maintain proper shade management.",
            ],
        },
    ],
    "cardamom": [
        {
            "name": "Capsule Rot (Phytophthora meadii)",
            "probability": 0.40, "severity": "high", "season": "Monsoon",
            "prevention": [
                "Spray Copper oxychloride at onset of monsoon.",
                "Improve drainage and reduce shade density.",
                "Remove and destroy infected panicles.",
            ],
        },
        {
            "name": "Thrips (Sciothrips cardamomi)",
            "probability": 0.50, "severity": "medium", "season": "Year-round",
            "prevention": [
                "Spray Dimethoate at panicle initiation.",
                "Maintain optimum shade (40-60%).",
                "Remove weeds that harbor thrips.",
            ],
        },
    ],
    "arecanut": [
        {
            "name": "Koleroga / Fruit Rot (Phytophthora arecae)",
            "probability": 0.45, "severity": "high", "season": "Monsoon",
            "prevention": [
                "Spray Bordeaux mixture (1%) before and during monsoon.",
                "Collect and destroy fallen diseased nuts.",
                "Ensure drainage around palm base.",
            ],
        },
        {
            "name": "Yellow Leaf Disease (Phytoplasma)",
            "probability": 0.30, "severity": "high", "season": "Year-round",
            "prevention": [
                "Remove and destroy severely infected palms.",
                "Control leafhopper vectors with Imidacloprid.",
                "Apply balanced nutrition with micronutrients.",
            ],
        },
    ],
    "cashewnut": [
        {
            "name": "Tea Mosquito Bug (Helopeltis antonii)",
            "probability": 0.55, "severity": "high", "season": "Flowering",
            "prevention": [
                "Spray Carbaryl or Lambda-cyhalothrin at flushing stage.",
                "Maintain canopy hygiene; prune overcrowded branches.",
                "Conserve natural enemies (Oecophylla smaragdina ants).",
            ],
        },
        {
            "name": "Anthracnose (Colletotrichum gloeosporioides)",
            "probability": 0.35, "severity": "medium", "season": "Monsoon",
            "prevention": [
                "Spray Carbendazim at new flush emergence.",
                "Prune and burn infected twigs.",
            ],
        },
    ],
    "cowpea": [
        {
            "name": "Mosaic Virus (CpMV)",
            "probability": 0.35, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Use virus-free seed from reliable sources.",
                "Control aphid vectors with Imidacloprid.",
                "Remove and destroy infected plants.",
            ],
        },
        {
            "name": "Pod Borer (Maruca vitrata)",
            "probability": 0.40, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Spray Emamectin benzoate at flowering.",
                "Use pheromone traps for monitoring.",
                "Intercrop with sorghum as barrier crop.",
            ],
        },
    ],
    "peas": [
        {
            "name": "Powdery Mildew (Erysiphe pisi)",
            "probability": 0.50, "severity": "medium", "season": "Rabi",
            "prevention": [
                "Spray Wettable Sulphur or Karathane at first sign.",
                "Use resistant varieties (Arkel, Azad P-1).",
                "Early sowing to escape late-season humidity.",
            ],
        },
        {
            "name": "Fusarium Wilt (Fusarium oxysporum f.sp. pisi)",
            "probability": 0.30, "severity": "medium", "season": "Rabi",
            "prevention": [
                "Seed treatment with Trichoderma + Carbendazim.",
                "Crop rotation with cereals for 3 years.",
            ],
        },
    ],
    "tobacco": [
        {
            "name": "Tobacco Mosaic Virus (TMV)",
            "probability": 0.40, "severity": "high", "season": "Kharif",
            "prevention": [
                "Use TMV-resistant varieties.",
                "Workers should wash hands before handling plants.",
                "Remove infected plants immediately to reduce spread.",
            ],
        },
        {
            "name": "Black Shank (Phytophthora nicotianae)",
            "probability": 0.35, "severity": "high", "season": "Kharif",
            "prevention": [
                "Use resistant varieties.",
                "Soil fumigation with Metam sodium.",
                "Improve field drainage; avoid waterlogging.",
            ],
        },
    ],
    "sweet potato": [
        {
            "name": "Sweet Potato Weevil (Cylas formicarius)",
            "probability": 0.50, "severity": "high", "season": "Year-round",
            "prevention": [
                "Plant weevil-free vine cuttings.",
                "Earth up vines to cover exposed tubers.",
                "Use pheromone traps for monitoring.",
            ],
        },
        {
            "name": "Scab (Elsinoe batatas)",
            "probability": 0.25, "severity": "low", "season": "Kharif",
            "prevention": [
                "Use disease-free planting material.",
                "Spray Mancozeb at vine stage.",
            ],
        },
    ],
    "tapioca": [
        {
            "name": "Cassava Mosaic Virus (CMV)",
            "probability": 0.40, "severity": "high", "season": "Year-round",
            "prevention": [
                "Use mosaic-free stem cuttings from certified sources.",
                "Control whitefly vectors with neem oil or Imidacloprid.",
                "Remove and destroy infected plants.",
            ],
        },
        {
            "name": "Tuber Rot (Phytophthora spp.)",
            "probability": 0.30, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Ensure good field drainage.",
                "Harvest at right maturity; avoid mechanical damage.",
            ],
        },
    ],
    "mesta": [
        {
            "name": "Stem Rot (Macrophomina phaseolina)",
            "probability": 0.35, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Seed treatment with Thiram + Carbendazim.",
                "Avoid water stagnation.",
                "Deep ploughing to bury inoculum.",
            ],
        },
        {
            "name": "Spiral Borer (Agrilus acutus)",
            "probability": 0.30, "severity": "medium", "season": "Kharif",
            "prevention": [
                "Early sowing in March-April to avoid peak borer.",
                "Collect and destroy infested stems.",
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
