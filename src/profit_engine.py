"""
Profit engine: compute production, revenue, cost, and profit per crop.

All monetary values are in Indian Rupees (₹).
Land area is internally handled in acres; the caller passes acres directly
(conversion from bigha happens in region_data_loader.bigha_to_acres).

Yield adjustment:
    effective_yield = region_avg_yield × (BASE_FACTOR + CONF_FACTOR × suitability_conf)

    This means:
      - suitability_conf = 1.0 → full regional yield
      - suitability_conf = 0.0 → 60% of regional yield (minimum conservative estimate)
"""

from src.config import YIELD_BASE_FACTOR, YIELD_CONF_FACTOR


def compute_profit(
    crop: str,
    region_context: dict,
    land_size_acres: float,
    suitability_conf: float,
) -> dict:
    """
    Compute economic output for a single crop given regional data and land size.

    Parameters
    ----------
    crop : str
        Normalised crop name (e.g. 'rice').
    region_context : dict
        Output of region_data_loader.get_region_context(). Must contain:
            yield_q_per_acre, price_per_quintal, cost_per_acre, data_confidence.
    land_size_acres : float
        Farmer's land area in acres (already converted from bigha).
    suitability_conf : float
        ML confidence score for this crop (0.0–1.0).

    Returns
    -------
    dict with keys:
        effective_yield_q_per_acre  : adjusted yield
        total_production_quintals   : yield × land_size
        price_per_quintal           : market price used
        gross_revenue_inr           : total revenue (₹)
        input_cost_inr              : total input cost (₹)
        net_profit_inr              : revenue − cost (₹)
        profit_per_acre_inr         : net_profit / land_size
        roi_pct                     : return on investment %
        data_confidence             : confidence level of region data
    """
    conf = max(0.0, min(1.0, float(suitability_conf)))
    land = max(0.0, float(land_size_acres))

    base_yield   = float(region_context["yield_q_per_acre"])
    price        = float(region_context["price_per_quintal"])
    cost_per_acre = float(region_context["cost_per_acre"])

    effective_yield   = base_yield * (YIELD_BASE_FACTOR + YIELD_CONF_FACTOR * conf)
    total_production  = round(effective_yield * land, 2)
    gross_revenue     = round(total_production * price, 0)
    total_cost        = round(cost_per_acre * land, 0)
    net_profit        = round(gross_revenue - total_cost, 0)
    profit_per_acre   = round(net_profit / land, 0) if land > 0 else 0.0
    roi_pct           = round((net_profit / total_cost * 100), 1) if total_cost > 0 else 0.0

    return {
        "effective_yield_q_per_acre": round(effective_yield, 2),
        "total_production_quintals":  total_production,
        "price_per_quintal":          price,
        "gross_revenue_inr":          gross_revenue,
        "input_cost_inr":             total_cost,
        "net_profit_inr":             net_profit,
        "profit_per_acre_inr":        profit_per_acre,
        "roi_pct":                    roi_pct,
        "data_confidence":            region_context.get("data_confidence", "fallback"),
    }


def normalise_profit_scores(crop_profits: list[dict]) -> list[dict]:
    """
    Add a normalised profit score (0–1) to each crop dict for balanced scoring.
    crop_profits: list of dicts, each must have 'net_profit_inr'.
    Returns the same list with 'profit_score_norm' added in-place.
    """
    profits = [c["net_profit_inr"] for c in crop_profits]
    min_p, max_p = min(profits), max(profits)
    span = max_p - min_p if max_p != min_p else 1.0
    for c in crop_profits:
        c["profit_score_norm"] = round((c["net_profit_inr"] - min_p) / span, 4)
    return crop_profits


def rank_by_profit(crop_list: list[dict]) -> list[dict]:
    """
    Sort crop dicts by net_profit_inr descending.
    Assigns 'rank' (1 = most profitable).
    """
    ranked = sorted(crop_list, key=lambda x: x.get("net_profit_inr", 0), reverse=True)
    for i, item in enumerate(ranked, 1):
        item["rank"] = i
    return ranked
