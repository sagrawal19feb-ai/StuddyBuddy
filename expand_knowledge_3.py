"""
expand_knowledge_3.py — Dev utility: dense structured knowledge expansion
=========================================================================
Generates thousands of accurate facts from compact structured tables
(countries, elements, formulas, units, people) using question templates.
Idempotent. DEV-ONLY — not part of the runtime app.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KB = ROOT / "data" / "knowledge.json"


def load_existing() -> set[str]:
    data = json.loads(KB.read_text(encoding="utf-8"))
    return {f.get("question", "").strip().lower() for f in data.get("facts", [])}


def add_facts(existing: set[str], facts: list[dict]) -> int:
    added = 0
    for fact in facts:
        if fact["question"].lower() in existing:
            continue
        existing.add(fact["question"].lower())
        facts_storage.append(fact)
        added += 1
    return added


facts_storage: list[dict] = []


def fact(category: str, question: str, answer: str, keywords: list[str]) -> dict:
    return {"question": question, "answer": answer, "category": category, "keywords": keywords}


# ============================================================================
# 1. COUNTRIES: name -> (capital, continent)
# ============================================================================
COUNTRIES: dict[str, tuple[str, str]] = {
    # Asia
    "Afghanistan": ("Kabul", "Asia"), "Armenia": ("Yerevan", "Asia"),
    "Azerbaijan": ("Baku", "Asia"), "Bahrain": ("Manama", "Asia"),
    "Bangladesh": ("Dhaka", "Asia"), "Bhutan": ("Thimphu", "Asia"),
    "Brunei": ("Bandar Seri Begawan", "Asia"), "Cambodia": ("Phnom Penh", "Asia"),
    "China": ("Beijing", "Asia"), "Cyprus": ("Nicosia", "Asia"),
    "Georgia": ("Tbilisi", "Asia"), "India": ("New Delhi", "Asia"),
    "Indonesia": ("Jakarta", "Asia"), "Iran": ("Tehran", "Asia"),
    "Iraq": ("Baghdad", "Asia"), "Israel": ("Jerusalem", "Asia"),
    "Japan": ("Tokyo", "Asia"), "Jordan": ("Amman", "Asia"),
    "Kazakhstan": ("Astana", "Asia"), "Kuwait": ("Kuwait City", "Asia"),
    "Kyrgyzstan": ("Bishkek", "Asia"), "Laos": ("Vientiane", "Asia"),
    "Lebanon": ("Beirut", "Asia"), "Malaysia": ("Kuala Lumpur", "Asia"),
    "Maldives": ("Malé", "Asia"), "Mongolia": ("Ulaanbaatar", "Asia"),
    "Myanmar": ("Naypyidaw", "Asia"), "Nepal": ("Kathmandu", "Asia"),
    "North Korea": ("Pyongyang", "Asia"), "Oman": ("Muscat", "Asia"),
    "Pakistan": ("Islamabad", "Asia"), "Palestine": ("Ramallah", "Asia"),
    "Philippines": ("Manila", "Asia"), "Qatar": ("Doha", "Asia"),
    "Saudi Arabia": ("Riyadh", "Asia"), "Singapore": ("Singapore", "Asia"),
    "South Korea": ("Seoul", "Asia"), "Sri Lanka": ("Colombo", "Asia"),
    "Syria": ("Damascus", "Asia"), "Taiwan": ("Taipei", "Asia"),
    "Tajikistan": ("Dushanbe", "Asia"), "Thailand": ("Bangkok", "Asia"),
    "Timor-Leste": ("Dili", "Asia"), "Turkey": ("Ankara", "Asia"),
    "Turkmenistan": ("Ashgabat", "Asia"), "United Arab Emirates": ("Abu Dhabi", "Asia"),
    "Uzbekistan": ("Tashkent", "Asia"), "Vietnam": ("Hanoi", "Asia"),
    "Yemen": ("Sanaa", "Asia"),
    # Europe
    "Albania": ("Tirana", "Europe"), "Andorra": ("Andorra la Vella", "Europe"),
    "Austria": ("Vienna", "Europe"), "Belarus": ("Minsk", "Europe"),
    "Belgium": ("Brussels", "Europe"), "Bosnia and Herzegovina": ("Sarajevo", "Europe"),
    "Bulgaria": ("Sofia", "Europe"), "Croatia": ("Zagreb", "Europe"),
    "Czech Republic": ("Prague", "Europe"), "Denmark": ("Copenhagen", "Europe"),
    "Estonia": ("Tallinn", "Europe"), "Finland": ("Helsinki", "Europe"),
    "France": ("Paris", "Europe"), "Germany": ("Berlin", "Europe"),
    "Greece": ("Athens", "Europe"), "Hungary": ("Budapest", "Europe"),
    "Iceland": ("Reykjavik", "Europe"), "Ireland": ("Dublin", "Europe"),
    "Italy": ("Rome", "Europe"), "Latvia": ("Riga", "Europe"),
    "Lithuania": ("Vilnius", "Europe"), "Luxembourg": ("Luxembourg City", "Europe"),
    "Malta": ("Valletta", "Europe"), "Moldova": ("Chișinău", "Europe"),
    "Monaco": ("Monaco", "Europe"), "Montenegro": ("Podgorica", "Europe"),
    "Netherlands": ("Amsterdam", "Europe"), "North Macedonia": ("Skopje", "Europe"),
    "Norway": ("Oslo", "Europe"), "Poland": ("Warsaw", "Europe"),
    "Portugal": ("Lisbon", "Europe"), "Romania": ("Bucharest", "Europe"),
    "Russia": ("Moscow", "Europe"), "Serbia": ("Belgrade", "Europe"),
    "Slovakia": ("Bratislava", "Europe"), "Slovenia": ("Ljubljana", "Europe"),
    "Spain": ("Madrid", "Europe"), "Sweden": ("Stockholm", "Europe"),
    "Switzerland": ("Bern", "Europe"), "Ukraine": ("Kyiv", "Europe"),
    "United Kingdom": ("London", "Europe"), "Vatican City": ("Vatican City", "Europe"),
    # Africa
    "Algeria": ("Algiers", "Africa"), "Angola": ("Luanda", "Africa"),
    "Benin": ("Porto-Novo", "Africa"), "Botswana": ("Gaborone", "Africa"),
    "Burkina Faso": ("Ouagadougou", "Africa"), "Burundi": ("Gitega", "Africa"),
    "Cameroon": ("Yaoundé", "Africa"), "Chad": ("N'Djamena", "Africa"),
    "Democratic Republic of the Congo": ("Kinshasa", "Africa"),
    "Republic of the Congo": ("Brazzaville", "Africa"),
    "Egypt": ("Cairo", "Africa"), "Ethiopia": ("Addis Ababa", "Africa"),
    "Gabon": ("Libreville", "Africa"), "Gambia": ("Banjul", "Africa"),
    "Ghana": ("Accra", "Africa"), "Guinea": ("Conakry", "Africa"),
    "Ivory Coast": ("Yamoussoukro", "Africa"), "Kenya": ("Nairobi", "Africa"),
    "Libya": ("Tripoli", "Africa"), "Madagascar": ("Antananarivo", "Africa"),
    "Malawi": ("Lilongwe", "Africa"), "Mali": ("Bamako", "Africa"),
    "Mauritania": ("Nouakchott", "Africa"), "Mauritius": ("Port Louis", "Africa"),
    "Morocco": ("Rabat", "Africa"), "Mozambique": ("Maputo", "Africa"),
    "Namibia": ("Windhoek", "Africa"), "Niger": ("Niamey", "Africa"),
    "Nigeria": ("Abuja", "Africa"), "Rwanda": ("Kigali", "Africa"),
    "Senegal": ("Dakar", "Africa"), "Sierra Leone": ("Freetown", "Africa"),
    "Somalia": ("Mogadishu", "Africa"), "South Africa": ("Pretoria", "Africa"),
    "South Sudan": ("Juba", "Africa"), "Sudan": ("Khartoum", "Africa"),
    "Tanzania": ("Dodoma", "Africa"), "Togo": ("Lomé", "Africa"),
    "Tunisia": ("Tunis", "Africa"), "Uganda": ("Kampala", "Africa"),
    "Zambia": ("Lusaka", "Africa"), "Zimbabwe": ("Harare", "Africa"),
    # Americas
    "Argentina": ("Buenos Aires", "South America"), "Bolivia": ("Sucre", "South America"),
    "Brazil": ("Brasília", "South America"), "Chile": ("Santiago", "South America"),
    "Colombia": ("Bogotá", "South America"), "Ecuador": ("Quito", "South America"),
    "Guyana": ("Georgetown", "South America"), "Paraguay": ("Asunción", "South America"),
    "Peru": ("Lima", "South America"), "Suriname": ("Paramaribo", "South America"),
    "Uruguay": ("Montevideo", "South America"), "Venezuela": ("Caracas", "South America"),
    "Canada": ("Ottawa", "North America"), "Mexico": ("Mexico City", "North America"),
    "United States": ("Washington, D.C.", "North America"),
    "Cuba": ("Havana", "North America"), "Jamaica": ("Kingston", "North America"),
    "Haiti": ("Port-au-Prince", "North America"), "Guatemala": ("Guatemala City", "North America"),
    "Honduras": ("Tegucigalpa", "North America"), "Nicaragua": ("Managua", "North America"),
    "Panama": ("Panama City", "North America"), "El Salvador": ("San Salvador", "North America"),
    "Costa Rica": ("San José", "North America"), "Belize": ("Belmopan", "North America"),
    "Dominican Republic": ("Santo Domingo", "North America"),
    "Trinidad and Tobago": ("Port of Spain", "North America"),
    "Bahamas": ("Nassau", "North America"), "Barbados": ("Bridgetown", "North America"),
    # Oceania
    "Australia": ("Canberra", "Oceania"), "Fiji": ("Suva", "Oceania"),
    "New Zealand": ("Wellington", "Oceania"), "Papua New Guinea": ("Port Moresby", "Oceania"),
    "Samoa": ("Apia", "Oceania"), "Solomon Islands": ("Honiara", "Oceania"),
    "Tonga": ("Nukuʻalofa", "Oceania"), "Vanuatu": ("Port Vila", "Oceania"),
    "Marshall Islands": ("Majuro", "Oceania"),
}

# ============================================================================
# 2. ELEMENTS: name -> (symbol, atomic_number, category)
# ============================================================================
ELEMENTS: dict[str, tuple[str, int, str]] = {
    "Hydrogen": ("H", 1, "nonmetal"), "Helium": ("He", 2, "noble gas"),
    "Lithium": ("Li", 3, "alkali metal"), "Beryllium": ("Be", 4, "alkaline earth metal"),
    "Boron": ("B", 5, "metalloid"), "Carbon": ("C", 6, "nonmetal"),
    "Nitrogen": ("N", 7, "nonmetal"), "Oxygen": ("O", 8, "nonmetal"),
    "Fluorine": ("F", 9, "halogen"), "Neon": ("Ne", 10, "noble gas"),
    "Sodium": ("Na", 11, "alkali metal"), "Magnesium": ("Mg", 12, "alkaline earth metal"),
    "Aluminium": ("Al", 13, "post-transition metal"), "Silicon": ("Si", 14, "metalloid"),
    "Phosphorus": ("P", 15, "nonmetal"), "Sulfur": ("S", 16, "nonmetal"),
    "Chlorine": ("Cl", 17, "halogen"), "Argon": ("Ar", 18, "noble gas"),
    "Potassium": ("K", 19, "alkali metal"), "Calcium": ("Ca", 20, "alkaline earth metal"),
    "Scandium": ("Sc", 21, "transition metal"), "Titanium": ("Ti", 22, "transition metal"),
    "Vanadium": ("V", 23, "transition metal"), "Chromium": ("Cr", 24, "transition metal"),
    "Manganese": ("Mn", 25, "transition metal"), "Iron": ("Fe", 26, "transition metal"),
    "Cobalt": ("Co", 27, "transition metal"), "Nickel": ("Ni", 28, "transition metal"),
    "Copper": ("Cu", 29, "transition metal"), "Zinc": ("Zn", 30, "transition metal"),
    "Gallium": ("Ga", 31, "post-transition metal"), "Germanium": ("Ge", 32, "metalloid"),
    "Arsenic": ("As", 33, "metalloid"), "Selenium": ("Se", 34, "nonmetal"),
    "Bromine": ("Br", 35, "halogen"), "Krypton": ("Kr", 36, "noble gas"),
    "Rubidium": ("Rb", 37, "alkali metal"), "Strontium": ("Sr", 38, "alkaline earth metal"),
    "Yttrium": ("Y", 39, "transition metal"), "Zirconium": ("Zr", 40, "transition metal"),
    "Niobium": ("Nb", 41, "transition metal"), "Molybdenum": ("Mo", 42, "transition metal"),
    "Technetium": ("Tc", 43, "transition metal"), "Ruthenium": ("Ru", 44, "transition metal"),
    "Rhodium": ("Rh", 45, "transition metal"), "Palladium": ("Pd", 46, "transition metal"),
    "Silver": ("Ag", 47, "transition metal"), "Cadmium": ("Cd", 48, "transition metal"),
    "Indium": ("In", 49, "post-transition metal"), "Tin": ("Sn", 50, "post-transition metal"),
    "Antimony": ("Sb", 51, "metalloid"), "Tellurium": ("Te", 52, "metalloid"),
    "Iodine": ("I", 53, "halogen"), "Xenon": ("Xe", 54, "noble gas"),
    "Cesium": ("Cs", 55, "alkali metal"), "Barium": ("Ba", 56, "alkaline earth metal"),
    "Lanthanum": ("La", 57, "lanthanide"), "Cerium": ("Ce", 58, "lanthanide"),
    "Praseodymium": ("Pr", 59, "lanthanide"), "Neodymium": ("Nd", 60, "lanthanide"),
    "Promethium": ("Pm", 61, "lanthanide"), "Samarium": ("Sm", 62, "lanthanide"),
    "Europium": ("Eu", 63, "lanthanide"), "Gadolinium": ("Gd", 64, "lanthanide"),
    "Terbium": ("Tb", 65, "lanthanide"), "Dysprosium": ("Dy", 66, "lanthanide"),
    "Holmium": ("Ho", 67, "lanthanide"), "Erbium": ("Er", 68, "lanthanide"),
    "Thulium": ("Tm", 69, "lanthanide"), "Ytterbium": ("Yb", 70, "lanthanide"),
    "Lutetium": ("Lu", 71, "lanthanide"), "Hafnium": ("Hf", 72, "transition metal"),
    "Tantalum": ("Ta", 73, "transition metal"), "Tungsten": ("W", 74, "transition metal"),
    "Rhenium": ("Re", 75, "transition metal"), "Osmium": ("Os", 76, "transition metal"),
    "Iridium": ("Ir", 77, "transition metal"), "Platinum": ("Pt", 78, "transition metal"),
    "Gold": ("Au", 79, "transition metal"), "Mercury": ("Hg", 80, "transition metal"),
    "Thallium": ("Tl", 81, "post-transition metal"), "Lead": ("Pb", 82, "post-transition metal"),
    "Bismuth": ("Bi", 83, "post-transition metal"), "Polonium": ("Po", 84, "post-transition metal"),
    "Astatine": ("At", 85, "halogen"), "Radon": ("Rn", 86, "noble gas"),
    "Francium": ("Fr", 87, "alkali metal"), "Radium": ("Ra", 88, "alkaline earth metal"),
    "Actinium": ("Ac", 89, "actinide"), "Thorium": ("Th", 90, "actinide"),
    "Protactinium": ("Pa", 91, "actinide"), "Uranium": ("U", 92, "actinide"),
    "Neptunium": ("Np", 93, "actinide"), "Plutonium": ("Pu", 94, "actinide"),
    "Americium": ("Am", 95, "actinide"), "Curium": ("Cm", 96, "actinide"),
    "Berkelium": ("Bk", 97, "actinide"), "Californium": ("Cf", 98, "actinide"),
    "Einsteinium": ("Es", 99, "actinide"), "Fermium": ("Fm", 100, "actinide"),
    "Mendelevium": ("Md", 101, "actinide"), "Nobelium": ("No", 102, "actinide"),
    "Lawrencium": ("Lr", 103, "actinide"), "Rutherfordium": ("Rf", 104, "transition metal"),
    "Dubnium": ("Db", 105, "transition metal"), "Seaborgium": ("Sg", 106, "transition metal"),
    "Bohrium": ("Bh", 107, "transition metal"), "Hassium": ("Hs", 108, "transition metal"),
    "Meitnerium": ("Mt", 109, "transition metal"),
}

# ============================================================================
# 3. MATH FORMULAS (topic, formula text, keywords)
# ============================================================================
MATH_FORMULAS: list[tuple[str, str, str]] = [
    ("area of a square", "side × side (s²)", "area square"),
    ("area of a rectangle", "length × breadth", "area rectangle"),
    ("area of a triangle", "½ × base × height", "area triangle"),
    ("area of a circle", "π × radius² (πr²)", "area circle"),
    ("circumference of a circle", "2 × π × radius (2πr)", "circumference circle"),
    ("perimeter of a square", "4 × side", "perimeter square"),
    ("perimeter of a rectangle", "2 × (length + breadth)", "perimeter rectangle"),
    ("volume of a cube", "side³", "volume cube"),
    ("volume of a cuboid", "length × breadth × height", "volume cuboid"),
    ("volume of a sphere", "(4/3) × π × radius³", "volume sphere"),
    ("surface area of a sphere", "4 × π × radius²", "surface area sphere"),
    ("volume of a cylinder", "π × radius² × height", "volume cylinder"),
    ("Pythagorean theorem", "a² + b² = c² for a right triangle", "pythagoras theorem"),
    ("quadratic formula", "x = (−b ± √(b² − 4ac)) / 2a", "quadratic formula"),
    ("distance formula", "distance = speed × time", "distance speed time"),
    ("speed formula", "speed = distance ÷ time", "speed formula"),
    ("simple interest", "SI = (P × R × T) / 100", "simple interest"),
    ("compound interest amount", "A = P × (1 + R/100)^T", "compound interest"),
    ("percentage", "percentage = (part ÷ whole) × 100", "percentage formula"),
    ("mean", "mean = sum of values ÷ number of values", "mean formula"),
    ("probability of an event", "favourable outcomes ÷ total outcomes", "probability formula"),
    ("slope of a line", "slope = (y₂ − y₁) / (x₂ − x₁)", "slope formula"),
    ("area of a trapezium", "½ × (sum of parallel sides) × height", "area trapezium"),
    ("area of a parallelogram", "base × height", "area parallelogram"),
    ("volume of a cone", "(1/3) × π × radius² × height", "volume cone"),
    ("volume of a pyramid", "(1/3) × base area × height", "volume pyramid"),
]

# ============================================================================
# 4. SI UNITS & PHYSICS (quantity, unit, extra)
# ============================================================================
SI_UNITS: list[tuple[str, str, str]] = [
    ("length", "metre (m)", "si unit length"),
    ("mass", "kilogram (kg)", "si unit mass"),
    ("time", "second (s)", "si unit time"),
    ("electric current", "ampere (A)", "si unit current"),
    ("temperature", "kelvin (K)", "si unit temperature"),
    ("amount of substance", "mole (mol)", "si unit mole"),
    ("luminous intensity", "candela (cd)", "si unit candela"),
    ("force", "newton (N)", "si unit force"),
    ("energy", "joule (J)", "si unit energy"),
    ("power", "watt (W)", "si unit power"),
    ("pressure", "pascal (Pa)", "si unit pressure"),
    ("frequency", "hertz (Hz)", "si unit frequency"),
    ("electric charge", "coulomb (C)", "si unit charge"),
    ("electric potential", "volt (V)", "si unit volt"),
    ("electric resistance", "ohm (Ω)", "si unit resistance"),
    ("capacitance", "farad (F)", "si unit capacitance"),
    ("magnetic flux", "weber (Wb)", "si unit magnetic flux"),
    ("area", "square metre (m²)", "si unit area"),
    ("volume", "cubic metre (m³)", "si unit volume"),
    ("velocity", "metre per second (m/s)", "si unit velocity"),
]

# ============================================================================
# 5. FAMOUS PEOPLE (name, achievement, era)
# ============================================================================
PEOPLE: list[tuple[str, str, str]] = [
    ("Albert Einstein", "developed the theory of relativity", "physicist"),
    ("Isaac Newton", "formulated the laws of motion and gravity", "physicist"),
    ("Marie Curie", "pioneered research on radioactivity", "scientist"),
    ("Charles Darwin", "proposed the theory of evolution by natural selection", "naturalist"),
    ("Nikola Tesla", "invented the alternating-current (AC) electrical system", "inventor"),
    ("Thomas Edison", "invented the practical electric light bulb", "inventor"),
    ("Galileo Galilei", "made pioneering telescope observations and supported heliocentrism", "astronomer"),
    ("Stephen Hawking", "theorised about black holes and cosmology", "physicist"),
    ("Michael Faraday", "discovered electromagnetic induction", "scientist"),
    ("James Watt", "improved the steam engine", "engineer"),
    ("Alexander Graham Bell", "invented the telephone", "inventor"),
    ("Guglielmo Marconi", "developed wireless telegraphy (radio)", "inventor"),
    ("Tim Berners-Lee", "invented the World Wide Web", "computer scientist"),
    ("Alan Turing", "laid the foundations of modern computing and AI", "computer scientist"),
    ("Bill Gates", "co-founded Microsoft", "entrepreneur"),
    ("Steve Jobs", "co-founded Apple", "entrepreneur"),
    ("Mark Zuckerberg", "co-founded Facebook", "entrepreneur"),
    ("Larry Page", "co-founded Google", "entrepreneur"),
    ("A.P.J. Abdul Kalam", "led India's missile and space programmes; 11th President of India", "scientist"),
    ("C.V. Raman", "discovered the Raman effect in light scattering", "physicist"),
    ("Homi J. Bhabha", "founded India's nuclear science programme", "physicist"),
    ("Vikram Sarabhai", "founded India's space programme (ISRO)", "scientist"),
    ("Srinivasa Ramanujan", "made extraordinary contributions to number theory", "mathematician"),
    ("Rabindranath Tagore", "wrote the Indian national anthem; first Asian Nobel laureate in literature", "poet"),
    ("Mother Teresa", "founded the Missionaries of Charity", "humanitarian"),
    ("Nelson Mandela", "fought apartheid and became South Africa's first Black president", "leader"),
    ("Martin Luther King Jr.", "led the American civil rights movement", "leader"),
    ("Winston Churchill", "led Britain during World War II", "leader"),
    ("Abraham Lincoln", "led the USA through the Civil War and ended slavery", "president"),
    ("Mahatma Gandhi", "led India's non-violent independence movement", "leader"),
    ("William Shakespeare", "wrote Romeo and Juliet and Hamlet", "playwright"),
    ("Jane Austen", "wrote Pride and Prejudice", "author"),
    ("Charles Dickens", "wrote A Tale of Two Cities and Oliver Twist", "author"),
    ("Mark Twain", "wrote The Adventures of Tom Sawyer", "author"),
    ("Leo Tolstoy", "wrote War and Peace", "author"),
    ("Sachin Tendulkar", "is cricket's highest run-scorer of all time", "sportsman"),
    ("M.S. Dhoni", "captained India to the 2011 Cricket World Cup", "cricketer"),
    ("Usain Bolt", "holds world records in the 100m and 200m sprints", "athlete"),
    ("Muhammad Ali", "was one of the greatest boxers in history", "boxer"),
    ("Michael Phelps", "won the most Olympic gold medals in history", "swimmer"),
    ("Malala Yousafzai", "is the youngest Nobel Peace Prize laureate, for girls' education", "activist"),
]

# ============================================================================
# 6. CURRENCIES for major countries (country, currency)
# ============================================================================
CURRENCIES: list[tuple[str, str]] = [
    ("India", "Indian Rupee (₹)"), ("United States", "US Dollar ($)"),
    ("United Kingdom", "Pound Sterling (£)"), ("Japan", "Japanese Yen (¥)"),
    ("China", "Chinese Yuan (¥)"), ("European Union", "Euro (€)"),
    ("Germany", "Euro (€)"), ("France", "Euro (€)"), ("Italy", "Euro (€)"),
    ("Spain", "Euro (€)"), ("Canada", "Canadian Dollar (C$)"),
    ("Australia", "Australian Dollar (A$)"), ("Brazil", "Brazilian Real (R$)"),
    ("Mexico", "Mexican Peso (Mex$)"), ("Russia", "Russian Ruble (₽)"),
    ("South Korea", "South Korean Won (₩)"), ("Switzerland", "Swiss Franc (CHF)"),
    ("Saudi Arabia", "Saudi Riyal (SAR)"), ("United Arab Emirates", "UAE Dirham (AED)"),
    ("Singapore", "Singapore Dollar (S$)"), ("Turkey", "Turkish Lira (₺)"),
    ("South Africa", "South African Rand (R)"), ("Egypt", "Egyptian Pound (EGP)"),
    ("Nigeria", "Nigerian Naira (₦)"), ("Pakistan", "Pakistani Rupee (PKR)"),
    ("Bangladesh", "Bangladeshi Taka (৳)"), ("Sri Lanka", "Sri Lankan Rupee (LKR)"),
    ("Nepal", "Nepalese Rupee (NPR)"), ("Afghanistan", "Afghan Afghani (AFN)"),
    ("Indonesia", "Indonesian Rupiah (IDR)"), ("Thailand", "Thai Baht (฿)"),
    ("Malaysia", "Malaysian Ringgit (RM)"), ("Philippines", "Philippine Peso (₱)"),
    ("Vietnam", "Vietnamese Dong (₫)"), ("Argentina", "Argentine Peso (ARS)"),
    ("Chile", "Chilean Peso (CLP)"), ("Colombia", "Colombian Peso (COP)"),
    ("Kenya", "Kenyan Shilling (KES)"), ("Ghana", "Ghanaian Cedi (GHS)"),
]


def main() -> None:
    global facts_storage
    data = json.loads(KB.read_text(encoding="utf-8"))
    facts_storage = data.setdefault("facts", [])
    existing = {f.get("question", "").strip().lower() for f in facts_storage}
    added = 0

    # -- countries: capital + continent --
    for country, (capital, continent) in COUNTRIES.items():
        added += add_facts(existing, [
            fact("Geography", f"What is the capital of {country}?",
                 f"The capital of {country} is {capital}.", [f"capital of {country.lower()}", capital.lower(), f"{country.lower()} capital"]),
            fact("Geography", f"Which continent is {country} in?",
                 f"{country} is in {continent}.", [f"{country.lower()} continent", f"where is {country.lower()}", continent.lower()]),
        ])

    # -- elements: symbol + atomic number --
    for name, (symbol, number, _cat) in ELEMENTS.items():
        added += add_facts(existing, [
            fact("Chemistry", f"What is the chemical symbol of {name}?",
                 f"The chemical symbol of {name} is {symbol}.", [f"symbol of {name.lower()}", f"{name.lower()} symbol", symbol.lower()]),
            fact("Chemistry", f"What is the atomic number of {name}?",
                 f"{name} has the atomic number {number}.", [f"atomic number of {name.lower()}", f"{name.lower()} atomic number"]),
        ])

    # -- math formulas --
    for topic, formula, kw in MATH_FORMULAS:
        added += add_facts(existing, [
            fact("Mathematics", f"What is the formula for the {topic}?",
                 f"The formula for the {topic} is: {formula}.", [f"formula {kw}", f"{kw} formula", topic]),
        ])

    # -- SI units --
    for quantity, unit, kw in SI_UNITS:
        added += add_facts(existing, [
            fact("Physics", f"What is the SI unit of {quantity}?",
                 f"The SI unit of {quantity} is the {unit}.", [f"si unit {quantity}", f"unit of {quantity}", kw]),
        ])

    # -- people --
    for name, achievement, role in PEOPLE:
        added += add_facts(existing, [
            fact("General Knowledge", f"Who was {name}?",
                 f"{name} was a {role} who {achievement}.", [name.lower(), f"who is {name.lower()}", f"about {name.lower()}"]),
        ])

    # -- currencies --
    for country, currency in CURRENCIES:
        added += add_facts(existing, [
            fact("General Knowledge", f"What is the currency of {country}?",
                 f"The currency of {country} is the {currency}.", [f"currency of {country.lower()}", f"{country.lower()} currency", currency.lower().split()[0]]),
        ])

    KB.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"added {added} facts (total {len(facts_storage)})")


if __name__ == "__main__":
    main()
