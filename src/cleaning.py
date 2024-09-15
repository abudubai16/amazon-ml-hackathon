import re
from typing import Dict, Tuple

# Define your synonym map
synonym_map = {
    'gms': 'gram', 'g': 'gram', 'kg': 'kilogram', 'kilos': 'kilogram',
    'oz': 'ounce', 'lbs': 'pound', 'pounds': 'pound', 'ton': 'ton',
    'liters': 'litre', 'l': 'litre', 'cm': 'centimetre', 'm': 'metre',
    'mm': 'millimetre', 'in': 'inch', 'ft': 'foot', 'foot': 'foot',
    'yd': 'yard'
}

entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

allowed_units = {unit for entity in entity_unit_map for unit in entity_unit_map[entity]}

def replace_synonyms(text: str) -> str:
    pattern = '|'.join(re.escape(key) for key in synonym_map.keys())
    replacement = lambda match: synonym_map.get(match.group(0).lower(), match.group(0))
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)

def extract_value_and_unit(text: str, entity_name: str) -> Tuple[str, str]:
    text = replace_synonyms(text)  # Apply synonym replacement
    patterns = {
        'width': r"(\d+(\.\d+)?)\s*(centimetre|foot|inch|metre|millimetre|yard)",
        'depth': r"(\d+(\.\d+)?)\s*(centimetre|foot|inch|metre|millimetre|yard)",
        'height': r"(\d+(\.\d+)?)\s*(centimetre|foot|inch|metre|millimetre|yard)",
        'item_weight': r"(\d+(\.\d+)?)\s*(gram|kilogram|microgram|milligram|ounce|pound|ton)",
        'maximum_weight_recommendation': r"(\d+(\.\d+)?)\s*(gram|kilogram|microgram|milligram|ounce|pound|ton)",
        'voltage': r"(\d+(\.\d+)?)\s*(kilovolt|millivolt|volt)",
        'wattage': r"(\d+(\.\d+)?)\s*(kilowatt|watt)",
        'item_volume': r"(\d+(\.\d+)?)\s*(centilitre|cubic foot|cubic inch|cup|decilitre|fluid ounce|gallon|imperial gallon|litre|microlitre|millilitre|pint|quart)"
    }
    
    pattern = patterns.get(entity_name, "")
    match = re.search(pattern, text.lower())
    
    if match:
        value, unit = match.group(1), match.group(3)
        unit = normalize_unit(unit, entity_name)
        return value, unit
    return "", ""

def normalize_unit(unit: str, entity_name: str) -> str:
    normalized_units = {
        'item_weight': {'gram': 'gram', 'kilogram': 'kilogram', 'microgram': 'microgram', 'milligram': 'milligram', 'ounce': 'ounce', 'pound': 'pound', 'ton': 'ton'},
        'maximum_weight_recommendation': {'gram': 'gram', 'kilogram': 'kilogram', 'microgram': 'microgram', 'milligram': 'milligram', 'ounce': 'ounce', 'pound': 'pound', 'ton': 'ton'},
        'voltage': {'kilovolt': 'kilovolt', 'millivolt': 'millivolt', 'volt': 'volt'},
        'wattage': {'kilowatt': 'kilowatt', 'watt': 'watt'},
        'item_volume': {'centilitre': 'centilitre', 'cubic foot': 'cubic foot', 'cubic inch': 'cubic inch', 'cup': 'cup', 'decilitre': 'decilitre', 'fluid ounce': 'fluid ounce', 'gallon': 'gallon', 'imperial gallon': 'imperial gallon', 'litre': 'litre', 'microlitre': 'microlitre', 'millilitre': 'millilitre', 'pint': 'pint', 'quart': 'quart'},
        'width': {'centimetre': 'centimetre', 'foot': 'foot', 'inch': 'inch', 'metre': 'metre', 'millimetre': 'millimetre', 'yard': 'yard'},
        'depth': {'centimetre': 'centimetre', 'foot': 'foot', 'inch': 'inch', 'metre': 'metre', 'millimetre': 'millimetre', 'yard': 'yard'},
        'height': {'centimetre': 'centimetre', 'foot': 'foot', 'inch': 'inch', 'metre': 'metre', 'millimetre': 'millimetre', 'yard': 'yard'}
    }
    
    return normalized_units.get(entity_name, {}).get(unit, "")

def clean_text(text: str, entity_name: str) -> str:
    value, unit = extract_value_and_unit(text, entity_name)
    if value and unit:
        return f"{value} {unit}"
    return ""
