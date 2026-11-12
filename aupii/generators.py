"""Level 1 - generators for synthetic Australian identifiers.

Design ideas worth noticing:

1. Every generator uses its own seeded random.Random, never the global
   `random`. Same seed -> same dataset, which makes experiments
   reproducible.
2. Valid checksum numbers are built by choosing the free digits at random
   and then SOLVING for the check digit(s).
3. The same number appears in several surface styles ("123456782",
   "123 456 782", "123-456-782"), because real documents are messy and the
   model must learn all of them.
4. `invalid()` produces hard negatives: right format, wrong checksum.
   These teach the model that "9 digits" is not the same as "a TFN".

Ethics note: a randomly generated number with a valid checksum can
coincide with a real person's identifier. Treat the dataset as
sensitive-looking data. Don't publish it as if it were "real", and say
in the README that it is synthetic.
"""

from __future__ import annotations

import random
import re
import unicodedata
from datetime import date, timedelta
from typing import Callable, Optional, Sequence

from . import validators as v

# ----------------------------------------------------------------------
# Vocabularies. build_dataset.py splits names, suburbs and street names
# 70/15/15 so val/test use values never seen in training.
# ----------------------------------------------------------------------

# Reflects Australia's mix: Anglo-Celtic, Chinese, Vietnamese, South Asian,
# Middle Eastern, European, Korean/Japanese, Filipino, Pacific, African.
# Month names (April, May, June) are left out on purpose: they would clash
# with dates like "3 May 1990".
FIRST_NAMES = (
    # Anglo-Celtic / common Australian
    "Olivia", "Jack", "Charlotte", "Noah", "Amelia", "William", "Isla", "Oliver",
    "Mia", "Lucas", "Hannah", "Liam", "Grace", "Ethan", "Chloe", "Daniel", "James",
    "Thomas", "Henry", "Leo", "Harrison", "Archie", "Hudson", "Lachlan", "Cooper",
    "Mitchell", "Bailey", "Jayden", "Riley", "Matilda", "Ava", "Ruby", "Sienna",
    "Harper", "Evie", "Georgia", "Madison", "Emily", "Jessica", "Sarah", "Rebecca",
    "Kylie", "Shane", "Darren", "Craig", "Bronwyn", "Sharon", "Gary", "Brett",
    "Kerry", "Lauren", "Nathan", "Joshua", "Samuel", "Benjamin", "Isabella",
    "Sophie", "Ella", "Lily", "Matthew", "Andrew", "Michelle", "Nicole", "Megan",
    "Heather", "Douglas", "Fiona", "Duncan", "Callum", "Kate", "Jacinta", "Declan",
    "Siobhan", "Ciaran", "Aoife", "Rhys", "Bethan", "Dylan", "Angus", "Hamish",
    "Fergus", "Mackenzie",
    # Chinese
    "Wei", "Mei", "Jun", "Xiaoming", "Jian", "Yan", "Hui", "Ming", "Xin", "Hao",
    "Jing", "Lei", "Ying", "Zhiwei", "Yuting", "Kai",
    # Vietnamese
    "Minh", "Linh", "Thanh", "Huong", "Duc", "Anh", "Tuan", "Trang", "Phuong",
    "Hieu", "Quang", "Lan",
    # South Asian
    "Priya", "Arjun", "Ravi", "Anjali", "Rohan", "Aarav", "Ananya", "Vikram",
    "Deepa", "Sanjay", "Lakshmi", "Karthik", "Divya", "Harpreet", "Gurpreet",
    "Manpreet", "Navjot", "Rahul", "Sneha", "Meera", "Aditya", "Ishaan", "Kavya",
    "Srinivas", "Venkatesh", "Nisha", "Pooja", "Imran", "Sadia", "Tharindu",
    "Nuwan", "Dilani", "Kasun",
    # Middle Eastern / Turkish / Persian
    "Ahmed", "Fatima", "Omar", "Aisha", "Mohammed", "Yusuf", "Layla", "Zainab",
    "Khalid", "Mariam", "Bilal", "Noor", "Elif", "Mehmet", "Emre", "Darius",
    "Shirin", "Reza", "Leila", "Tariq",
    # European
    "Giorgos", "Eleni", "Nikos", "Dimitra", "Kostas", "Marco", "Sofia", "Giuseppe",
    "Lucia", "Alessandro", "Francesca", "Matteo", "Chiara", "Mateo", "José",
    "Carmen", "Pablo", "Inés", "Luca", "Stefan", "Katarina", "Marija", "Ivan",
    "Petra", "Tomasz", "Agnieszka", "Jakub", "Lukas", "Anna", "Hans", "Ingrid",
    "Lars", "Sven", "Renée", "Zoë",
    # Korean / Japanese
    "Hiroshi", "Yuki", "Haruto", "Sakura", "Kenji", "Aiko", "Ji-woo", "Min-jun",
    "Seo-yeon", "Hyun", "Jae",
    # Filipino
    "Maricel", "Rodel", "Jericho", "Analyn", "Rowena", "Joy",
    # Pacific / Māori
    "Kiri", "Tane", "Aroha", "Manaia", "Sione", "Losa", "Tevita", "Mele",
    # African
    "Kwame", "Amara", "Chidi", "Ngozi", "Abebe", "Tsion", "Kofi", "Adaeze", "Sipho",
)

LAST_NAMES = (
    # Anglo-Celtic
    "Smith", "Jones", "Williams", "Brown", "Wilson", "Taylor", "Johnson", "White",
    "Martin", "Anderson", "Thompson", "Walker", "Harris", "Clarke", "Robinson",
    "Wright", "Thomas", "Mitchell", "Campbell", "Kelly", "Murphy", "O'Brien",
    "O'Connor", "Ryan", "Sullivan", "McDonald", "MacKenzie", "Stewart", "Fraser",
    "Morrison", "Hughes", "Evans", "Davies", "Roberts", "Edwards", "Hall", "Young",
    "King", "Scott", "Green", "Baker", "Carter", "Turner", "Phillips", "Parker",
    "Cooper", "Bennett", "Ward", "Watson", "Brooks", "Reid", "Hamilton", "Graham",
    "Russell", "Gibson", "Ellis", "Price", "Webb", "Chapman", "Dawson", "Fletcher",
    "Hayes", "Lawson", "Marshall", "Palmer", "Barnes", "Henderson", "Cunningham",
    "Gallagher", "Doyle", "Fitzgerald", "Quinn", "Smith-Jones", "Ashworth-Hill",
    # Vietnamese
    "Nguyen", "Tran", "Le", "Pham", "Hoang", "Huynh", "Vo", "Dang", "Bui", "Do",
    "Ngo", "Duong", "Truong", "Vu",
    # Chinese (Mandarin and Cantonese romanisations)
    "Chen", "Wang", "Li", "Zhang", "Liu", "Yang", "Huang", "Zhao", "Wu", "Zhou",
    "Xu", "Sun", "Ma", "Zhu", "Hu", "Guo", "Lin", "He", "Luo", "Tan", "Lim", "Ng",
    "Wong", "Chan", "Cheung", "Lau", "Leung", "Ho", "Yip", "Tang",
    # Korean / Japanese
    "Kim", "Lee", "Park", "Choi", "Jung", "Kang", "Cho", "Yoon", "Tanaka",
    "Suzuki", "Sato", "Watanabe", "Yamamoto", "Nakamura", "Kobayashi",
    # South Asian
    "Singh", "Kaur", "Patel", "Sharma", "Kumar", "Gupta", "Shah", "Mehta", "Reddy",
    "Rao", "Iyer", "Nair", "Menon", "Pillai", "Krishnan", "Subramaniam",
    "Chowdhury", "Rahman", "Hossain", "Malik", "Qureshi", "Siddiqui", "Perera",
    "Fernando", "De Silva", "Jayasuriya", "Wickramasinghe", "Gill", "Sandhu",
    "Dhillon", "Grewal", "Bose", "Banerjee", "Das",
    # Middle Eastern / Turkish / Persian
    "Khan", "Ali", "Hassan", "Hussein", "Ibrahim", "Haddad", "Khoury", "Nasser",
    "Saleh", "Mansour", "Karimi", "Hosseini", "Yilmaz", "Demir", "Kaya", "Abdullah",
    # European
    "Papadopoulos", "Georgiou", "Nikolaidis", "Konstantinou", "Rossi", "Russo",
    "Ferrari", "Esposito", "Romano", "Bianchi", "Di Stefano", "De Luca", "Costa",
    "Garcia", "Rodriguez", "Martinez", "Lopez", "Fernandez", "Silva", "Santos",
    "Oliveira", "Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Novak",
    "Kowalski", "Nowak", "Horvat", "Petrovic", "Jovanovic", "Kovac", "Ivanov",
    "Popescu", "Van der Merwe", "Van Dijk", "De Jong", "Jansen", "Nielsen",
    "Hansen", "Andersson", "Lindqvist",
    # Filipino
    "Reyes", "Cruz", "Bautista", "Dela Cruz", "Mendoza", "Villanueva", "Ramos",
    "Aquino",
    # African
    "Mensah", "Okafor", "Adeyemi", "Mwangi", "Otieno", "Tesfaye", "Bekele",
    "Dlamini", "Nkosi", "Boateng", "Osei",
    # Pacific
    "Tupou", "Fifita", "Faleolo", "Ngata", "Taufa", "Leota",
)


def email_part(name: str) -> str:
    """'Van der Merwe' -> 'vandermerwe', 'Zoë' -> 'zoe', "O'Brien" -> 'obrien'.

    Emails are almost always plain ASCII letters, so strip accents
    (NFKD splits 'ë' into 'e' + a combining mark, which we drop), then
    spaces, hyphens and apostrophes.
    """
    ascii_only = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", ascii_only.lower())

EMAIL_DOMAINS = (
    "gmail.com", "outlook.com", "hotmail.com", "yahoo.com.au", "bigpond.com",
    "icloud.com", "example.com", "example.org",
)

# (suburb, state, postcode) - real suburb/postcode pairs, all 8 states and
# territories, cities and regional towns. build_dataset.py gives each split
# its own subset, so val/test addresses use suburbs never seen in training.
SUBURBS = (
    # VIC
    ("Melbourne", "VIC", "3000"), ("Carlton", "VIC", "3053"), ("Fitzroy", "VIC", "3065"),
    ("Collingwood", "VIC", "3066"), ("Richmond", "VIC", "3121"), ("Hawthorn", "VIC", "3122"),
    ("Kew", "VIC", "3101"), ("Box Hill", "VIC", "3128"), ("Doncaster", "VIC", "3108"),
    ("Footscray", "VIC", "3011"), ("Sunshine", "VIC", "3020"), ("St Kilda", "VIC", "3182"),
    ("Brighton", "VIC", "3186"), ("Prahran", "VIC", "3181"), ("South Yarra", "VIC", "3141"),
    ("Toorak", "VIC", "3142"), ("Glen Waverley", "VIC", "3150"), ("Clayton", "VIC", "3168"),
    ("Dandenong", "VIC", "3175"), ("Frankston", "VIC", "3199"), ("Cranbourne", "VIC", "3977"),
    ("Berwick", "VIC", "3806"), ("Pakenham", "VIC", "3810"), ("Werribee", "VIC", "3030"),
    ("Point Cook", "VIC", "3030"), ("Craigieburn", "VIC", "3064"), ("Epping", "VIC", "3076"),
    ("Reservoir", "VIC", "3073"), ("Preston", "VIC", "3072"), ("Brunswick", "VIC", "3056"),
    ("Coburg", "VIC", "3058"), ("Essendon", "VIC", "3040"), ("Northcote", "VIC", "3070"),
    ("Williamstown", "VIC", "3016"), ("Ringwood", "VIC", "3134"), ("Bundoora", "VIC", "3083"),
    ("Mount Waverley", "VIC", "3149"), ("Caulfield", "VIC", "3162"), ("Melton", "VIC", "3337"),
    ("Sunbury", "VIC", "3429"), ("Geelong", "VIC", "3220"), ("Ballarat", "VIC", "3350"),
    ("Bendigo", "VIC", "3550"), ("Mildura", "VIC", "3500"), ("Shepparton", "VIC", "3630"),
    ("Warrnambool", "VIC", "3280"), ("Traralgon", "VIC", "3844"),
    # NSW
    ("Sydney", "NSW", "2000"), ("Surry Hills", "NSW", "2010"), ("Newtown", "NSW", "2042"),
    ("Bondi", "NSW", "2026"), ("Bondi Junction", "NSW", "2022"), ("Manly", "NSW", "2095"),
    ("Chatswood", "NSW", "2067"), ("Parramatta", "NSW", "2150"), ("Blacktown", "NSW", "2148"),
    ("Penrith", "NSW", "2750"), ("Liverpool", "NSW", "2170"), ("Bankstown", "NSW", "2200"),
    ("Hurstville", "NSW", "2220"), ("Cronulla", "NSW", "2230"), ("Ryde", "NSW", "2112"),
    ("Hornsby", "NSW", "2077"), ("Castle Hill", "NSW", "2154"), ("Strathfield", "NSW", "2135"),
    ("Burwood", "NSW", "2134"), ("Marrickville", "NSW", "2204"), ("Randwick", "NSW", "2031"),
    ("Mosman", "NSW", "2088"), ("Campbelltown", "NSW", "2560"), ("Wollongong", "NSW", "2500"),
    ("Newcastle", "NSW", "2300"), ("Gosford", "NSW", "2250"), ("Wagga Wagga", "NSW", "2650"),
    ("Albury", "NSW", "2640"), ("Dubbo", "NSW", "2830"), ("Tamworth", "NSW", "2340"),
    ("Orange", "NSW", "2800"), ("Bathurst", "NSW", "2795"), ("Coffs Harbour", "NSW", "2450"),
    ("Port Macquarie", "NSW", "2444"), ("Lismore", "NSW", "2480"), ("Queanbeyan", "NSW", "2620"),
    # QLD
    ("Brisbane City", "QLD", "4000"), ("Fortitude Valley", "QLD", "4006"),
    ("South Brisbane", "QLD", "4101"), ("West End", "QLD", "4101"), ("Toowong", "QLD", "4066"),
    ("Indooroopilly", "QLD", "4068"), ("Chermside", "QLD", "4032"), ("Carindale", "QLD", "4152"),
    ("Ipswich", "QLD", "4305"), ("Logan Central", "QLD", "4114"), ("Redcliffe", "QLD", "4020"),
    ("Southport", "QLD", "4215"), ("Surfers Paradise", "QLD", "4217"), ("Robina", "QLD", "4226"),
    ("Burleigh Heads", "QLD", "4220"), ("Maroochydore", "QLD", "4558"),
    ("Noosa Heads", "QLD", "4567"), ("Caloundra", "QLD", "4551"), ("Toowoomba", "QLD", "4350"),
    ("Cairns", "QLD", "4870"), ("Townsville", "QLD", "4810"), ("Mackay", "QLD", "4740"),
    ("Rockhampton", "QLD", "4700"), ("Bundaberg", "QLD", "4670"), ("Hervey Bay", "QLD", "4655"),
    ("Gladstone", "QLD", "4680"),
    # SA
    ("Adelaide", "SA", "5000"), ("North Adelaide", "SA", "5006"), ("Glenelg", "SA", "5045"),
    ("Norwood", "SA", "5067"), ("Unley", "SA", "5061"), ("Prospect", "SA", "5082"),
    ("Salisbury", "SA", "5108"), ("Elizabeth", "SA", "5112"), ("Modbury", "SA", "5092"),
    ("Marion", "SA", "5043"), ("Port Adelaide", "SA", "5015"), ("Mount Barker", "SA", "5251"),
    ("Mount Gambier", "SA", "5290"), ("Whyalla", "SA", "5600"), ("Port Augusta", "SA", "5700"),
    ("Murray Bridge", "SA", "5253"),
    # WA
    ("Perth", "WA", "6000"), ("Northbridge", "WA", "6003"), ("Subiaco", "WA", "6008"),
    ("Fremantle", "WA", "6160"), ("Cottesloe", "WA", "6011"), ("Scarborough", "WA", "6019"),
    ("Joondalup", "WA", "6027"), ("Midland", "WA", "6056"), ("Armadale", "WA", "6112"),
    ("Rockingham", "WA", "6168"), ("Mandurah", "WA", "6210"), ("Bunbury", "WA", "6230"),
    ("Geraldton", "WA", "6530"), ("Kalgoorlie", "WA", "6430"), ("Albany", "WA", "6330"),
    ("Broome", "WA", "6725"), ("Victoria Park", "WA", "6100"), ("Cannington", "WA", "6107"),
    # TAS
    ("Hobart", "TAS", "7000"), ("Sandy Bay", "TAS", "7005"), ("Battery Point", "TAS", "7004"),
    ("Glenorchy", "TAS", "7010"), ("Kingston", "TAS", "7050"), ("Launceston", "TAS", "7250"),
    ("Devonport", "TAS", "7310"), ("Burnie", "TAS", "7320"), ("Ulverstone", "TAS", "7315"),
    # NT
    ("Darwin", "NT", "0800"), ("Palmerston", "NT", "0830"), ("Casuarina", "NT", "0810"),
    ("Nightcliff", "NT", "0810"), ("Alice Springs", "NT", "0870"), ("Katherine", "NT", "0850"),
    # ACT
    ("Canberra", "ACT", "2601"), ("Dickson", "ACT", "2602"), ("Griffith", "ACT", "2603"),
    ("Kingston", "ACT", "2604"), ("Phillip", "ACT", "2606"), ("Braddon", "ACT", "2612"),
    ("Turner", "ACT", "2612"), ("Belconnen", "ACT", "2617"), ("Greenway", "ACT", "2900"),
    ("Gungahlin", "ACT", "2912"),
)

STREET_NAMES = (
    "High", "Church", "Station", "Victoria", "Park", "George", "King", "Queen",
    "Bridge", "Railway", "Albert", "Elizabeth", "Wattle", "Banksia", "Acacia",
    "Hill", "Main", "Smith", "Ocean", "River", "Collins", "Bourke", "Flinders",
    "Swanston", "Lonsdale", "Chapel", "Glenferrie", "Burke", "Canterbury", "Whitehorse",
    "Pitt", "Oxford", "Crown", "Bay", "Beach", "Harbour", "Pacific", "Military",
    "Anzac", "Federation", "Hume", "Murray", "Darling", "Lachlan", "Macquarie",
    "Hunter", "Wellington", "Nelson", "Grey", "Stirling", "Hay", "Murray River",
    "Grenfell", "Rundle", "Hindley", "Waymouth", "Gouger", "Ann", "Adelaide",
    "Mary", "Margaret", "Charlotte", "Edward", "William", "Russell", "Spring",
    "Lygon", "Brunswick", "Sydney", "Nicholson", "Hoddle", "Punt", "Toorak",
    "Dandenong", "Warrigal", "Springvale", "Stud", "Blackburn", "Ferntree Gully",
    "Boronia", "Grevillea", "Jacaranda", "Ironbark", "Bottlebrush", "Kurrajong",
    "Myrtle", "Lilly Pilly", "Tea Tree", "Magpie", "Kookaburra", "Lyrebird",
    "Bellbird", "Rosella", "Wombat", "Koala", "Echidna", "Platypus", "Cockatoo",
    "Sunset", "Seaview", "Ridge", "Valley", "Forest", "Lakeside", "Highland",
    "Settlers", "Pioneer", "Heritage", "Bluegum", "Riverside", "Mountain",
)

# (full, abbreviated)
STREET_TYPES = (
    ("Street", "St"), ("Road", "Rd"), ("Avenue", "Ave"), ("Drive", "Dr"),
    ("Court", "Ct"), ("Place", "Pl"), ("Crescent", "Cres"), ("Lane", "Ln"),
    ("Parade", "Pde"), ("Way", "Way"), ("Boulevard", "Blvd"), ("Highway", "Hwy"),
    ("Terrace", "Tce"), ("Close", "Cl"), ("Grove", "Gr"), ("Circuit", "Cct"),
    ("Esplanade", "Esp"), ("Square", "Sq"), ("Rise", "Rise"), ("Track", "Trk"),
)

MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")

# Two-digit institution codes used by several Australian banks (first
# two digits of a BSB). The third digit is the state: 2 NSW/ACT, 3 VIC,
# 4 QLD, 5 SA/NT, 6 WA, 7 TAS.
BSB_INSTITUTION_CODES = ("01", "03", "06", "08", "12", "18", "48", "63", "73", "76")

# Letter prefixes seen on Australian passports (approximate).
PASSPORT_PREFIXES = ("N", "E", "M", "L", "P", "R", "PA", "PB", "PC", "PD", "PE", "RA")

# Digit-count ranges for the numeric licence formats (see validators.py).
LICENCE_LENGTHS = {"NSW": (6, 8), "VIC": (8, 10), "QLD": (8, 9), "WA": (7, 7),
                   "NT": (5, 10), "ACT": (6, 10)}

# Medicare issue numbers are mostly low (most people are on card 1-3).
_ISSUE_WEIGHTS = (30, 25, 15, 10, 7, 5, 4, 2, 2)


DATE_STYLES = ("dmy_slash", "dmy_slash", "dmy_short", "iso",
               "dmy_dash", "dmy_dot", "long", "abbrev")


def format_date(d: date, style: str) -> str:
    """Format a date in one of the styles common in Australian documents."""
    return {
        "dmy_slash": f"{d.day:02d}/{d.month:02d}/{d.year}",
        "dmy_short": f"{d.day}/{d.month}/{d.year}",
        "iso": d.isoformat(),
        "dmy_dash": f"{d.day:02d}-{d.month:02d}-{d.year}",
        "dmy_dot": f"{d.day:02d}.{d.month:02d}.{d.year}",
        "long": f"{d.day} {MONTHS[d.month - 1]} {d.year}",
        "abbrev": f"{d.day} {MONTHS[d.month - 1][:3]} {d.year}",
    }[style]


def _join(digits: Sequence[int]) -> str:
    return "".join(str(d) for d in digits)


def _group(text: str, sizes: Sequence[int], sep: str) -> str:
    """'123456782', (3,3,3), ' ' -> '123 456 782'."""
    parts, i = [], 0
    for size in sizes:
        parts.append(text[i:i + size])
        i += size
    if i < len(text):
        parts.append(text[i:])
    return sep.join(p for p in parts if p)


class PIIGenerator:
    """Seeded generator for every entity type in labels.py."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        # Name and address vocabularies. build_dataset.py swaps these per
        # split so val/test use names, suburbs and streets never seen in training.
        self.first_names: Sequence[str] = FIRST_NAMES
        self.last_names: Sequence[str] = LAST_NAMES
        self.suburbs: Sequence[tuple[str, str, str]] = SUBURBS
        self.street_names: Sequence[str] = STREET_NAMES

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _digits(self, n: int) -> list[int]:
        return [self.rng.randint(0, 9) for _ in range(n)]

    def _style(self, style: Optional[str], allowed: Sequence[str]) -> str:
        if style is None:
            return self.rng.choice(allowed)
        if style not in allowed:
            raise ValueError(f"style must be one of {allowed}, got {style!r}")
        return style

    @staticmethod
    def _format(number: str, groups: Sequence[int], style: str) -> str:
        if style == "plain":
            return number
        sep = {"spaced": " ", "hyphen": "-"}[style]
        return _group(number, groups, sep)

    # ------------------------------------------------------------------
    # checksum identifiers
    # ------------------------------------------------------------------

    def tfn(self, style: Optional[str] = None) -> str:
        """e.g. '123 456 782'. Styles: plain, spaced, hyphen."""
        style = self._style(style, ("plain", "spaced", "hyphen"))
        while True:
            body = self._digits(8)
            # Need (S + 10*c) % 11 == 0. Since 10 = -1 (mod 11), c = S mod 11.
            check = v.weighted_sum(body, v.TFN_WEIGHTS[:8]) % 11
            if check < 10 and any(body + [check]):   # 10 isn't a digit -> retry
                break
        return self._format(_join(body + [check]), (3, 3, 3), style)

    def abn(self, style: Optional[str] = None) -> str:
        """e.g. '51 824 753 556'. Styles: plain, spaced."""
        style = self._style(style, ("plain", "spaced"))
        tail = self._digits(9)
        tail_sum = v.weighted_sum(tail, v.ABN_WEIGHTS[2:])
        # The first two digits contribute 10*(a-1) + b, which covers 0..89,
        # so there is always at least one pair that makes the total % 89 == 0.
        pairs = [(a, b) for a in range(1, 10) for b in range(10)
                 if (tail_sum + 10 * (a - 1) + b) % 89 == 0]
        a, b = self.rng.choice(pairs)
        return self._format(_join([a, b] + tail), (2, 3, 3, 3), style)

    def acn(self, style: Optional[str] = None) -> str:
        """e.g. '004 085 616'. Styles: plain, spaced, hyphen."""
        style = self._style(style, ("plain", "spaced", "hyphen"))
        while True:
            body = self._digits(8)
            if any(body):
                break
        return self._format(_join(body + [v.acn_check_digit(body)]), (3, 3, 3), style)

    def medicare(self, style: Optional[str] = None, include_irn: bool = False) -> str:
        """e.g. '2123 45670 1'. Styles: plain, spaced, hyphen.

        With include_irn=True a 1-digit IRN is appended: '2123 45670 1/2'.
        """
        style = self._style(style, ("plain", "spaced", "hyphen"))
        body = [self.rng.randint(2, 6)] + self._digits(7)
        check = v.medicare_check_digit(body)
        issue = self.rng.choices(range(1, 10), weights=_ISSUE_WEIGHTS)[0]
        text = self._format(_join(body + [check, issue]), (4, 5, 1), style)
        if include_irn:
            irn = str(self.rng.randint(1, 9))
            sep = "" if style == "plain" else self.rng.choice(["/", " ", "-"])
            text += sep + irn
        return text

    # ------------------------------------------------------------------
    # format-only identifiers
    # ------------------------------------------------------------------

    def bsb(self, style: Optional[str] = None) -> str:
        """e.g. '063-123'. Styles: hyphen (most common), plain, spaced."""
        style = self._style(style, ("hyphen", "hyphen", "plain", "spaced"))
        code = self.rng.choice(BSB_INSTITUTION_CODES)
        number = code + str(self.rng.randint(2, 7)) + _join(self._digits(3))
        return self._format(number, (3, 3), style)

    def bank_account(self, style: Optional[str] = None) -> str:
        """6-9 digits. Styles: plain, spaced (groups of 4)."""
        style = self._style(style, ("plain", "plain", "spaced"))
        number = _join(self._digits(self.rng.randint(6, 9)))
        groups = {8: (4, 4), 9: (3, 3, 3), 6: (3, 3)}.get(len(number))
        if groups is None:          # 7 digits: no natural grouping
            return number
        return self._format(number, groups, style)

    def passport(self) -> str:
        """e.g. 'PA1234567'."""
        return self.rng.choice(PASSPORT_PREFIXES) + _join(self._digits(7))

    def driver_licence(self, state: Optional[str] = None) -> str:
        """Licence number in one state's (approximate) format."""
        state = state or self.rng.choice(list(v.DRIVER_LICENCE_PATTERNS))
        if state in ("SA", "TAS"):
            return self.rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") + _join(self._digits(5))
        lo, hi = LICENCE_LENGTHS[state]
        return _join(self._digits(self.rng.randint(lo, hi)))

    def phone(self, kind: Optional[str] = None, style: Optional[str] = None,
              fictional: bool = False) -> str:
        """Mobile or landline.

        kind: 'mobile' or 'landline' (random if None, 60/40)
        mobile styles:   plain '0412345678', spaced '0412 345 678',
                         intl '+61 412 345 678'
        landline styles: plain '0398765432', spaced '03 9876 5432',
                         paren '(03) 9876 5432', intl '+61 3 9876 5432'
        fictional=True (landline only) uses the 5550 xxxx block that ACMA
        sets aside for use in fiction, so it can never ring a real person.
        """
        if kind is None:
            kind = "mobile" if self.rng.random() < 0.6 else "landline"

        if kind == "mobile":
            style = self._style(style, ("plain", "spaced", "intl"))
            rest = _join(self._digits(8))                     # 4 + 8 digits
            if style == "plain":
                return "04" + rest
            if style == "spaced":
                return f"04{rest[:2]} {rest[2:5]} {rest[5:]}"
            return f"+61 4{rest[:2]} {rest[2:5]} {rest[5:]}"

        if kind != "landline":
            raise ValueError("kind must be 'mobile' or 'landline'")
        style = self._style(style, ("plain", "spaced", "paren", "intl"))
        area = self.rng.choice("2378")
        if fictional:
            local = "5550" + _join(self._digits(4))
        else:
            local = str(self.rng.randint(2, 9)) + _join(self._digits(7))
        if style == "plain":
            return f"0{area}{local}"
        if style == "spaced":
            return f"0{area} {local[:4]} {local[4:]}"
        if style == "paren":
            return f"(0{area}) {local[:4]} {local[4:]}"
        return f"+61 {area} {local[:4]} {local[4:]}"

    # ------------------------------------------------------------------
    # general PII
    # ------------------------------------------------------------------

    def name_parts(self) -> tuple[str, str]:
        """A random (first, last) pair. Reuse it to keep a document consistent."""
        return self.rng.choice(self.first_names), self.rng.choice(self.last_names)

    def person_name(self, style: Optional[str] = None,
                    first: Optional[str] = None, last: Optional[str] = None) -> str:
        """Styles: full 'Priya Nguyen', initial 'Priya K. Nguyen',
        surname_first 'NGUYEN, Priya'. Pass first/last to name a known person."""
        style = self._style(style, ("full", "full", "initial", "surname_first"))
        if first is None or last is None:
            first, last = self.name_parts()
        if style == "full":
            return f"{first} {last}"
        if style == "initial":
            return f"{first} {self.rng.choice('ABCDEFGHJKLMNPRSTW')}. {last}"
        return f"{last.upper()}, {first}"

    def email(self, first: Optional[str] = None, last: Optional[str] = None) -> str:
        """Built from a name so it can match a PERSON in the same document."""
        if first is None or last is None:
            first, last = self.name_parts()
        f, l = email_part(first), email_part(last)
        local = self.rng.choice([f"{f}.{l}", f"{f}{l}", f"{f[0]}.{l}", f"{f}_{l}", f"{l}{f[0]}"])
        if self.rng.random() < 0.3:
            local += str(self.rng.randint(1, 99))
        return f"{local}@{self.rng.choice(EMAIL_DOMAINS)}"

    def date_of_birth(self, style: Optional[str] = None) -> str:
        """Random DOB 1940-2008. Styles: dmy_slash '03/04/1990',
        dmy_short '3/4/1990', iso '1990-04-03', dmy_dash '03-04-1990',
        dmy_dot '03.04.1990', long '3 April 1990', abbrev '3 Apr 1990'."""
        return format_date(self.random_date(date(1940, 1, 1), date(2008, 12, 31)),
                           self._style(style, DATE_STYLES))

    def random_date(self, start: date, end: date) -> date:
        return start + timedelta(days=self.rng.randint(0, (end - start).days))

    def address(self, style: Optional[str] = None) -> str:
        """Styles: one_line '12 Smith Street, Richmond VIC 3121',
        multi_line (newline before suburb), comma '12 Smith St, Richmond, VIC, 3121'."""
        style = self._style(style, ("one_line", "one_line", "multi_line", "comma"))
        suburb, state, postcode = self.rng.choice(self.suburbs)
        full, abbrev = self.rng.choice(STREET_TYPES)
        street_type = full if self.rng.random() < 0.6 else abbrev
        number = str(self.rng.randint(1, 250))
        if self.rng.random() < 0.08:                                  # 12A, 7B
            number += self.rng.choice("ABC")
        street = f"{number} {self.rng.choice(self.street_names)} {street_type}"
        if self.rng.random() < 0.25:                                  # unit / level
            unit = self.rng.randint(1, 40)
            street = self.rng.choice([f"Unit {unit}, {street}", f"{unit}/{street}",
                                      f"Apt {unit}, {street}", f"Level {unit % 20 + 1}, {street}"])
        if style == "one_line":
            return f"{street}, {suburb} {state} {postcode}"
        if style == "multi_line":
            return f"{street}\n{suburb} {state} {postcode}"
        return f"{street}, {suburb}, {state}, {postcode}"

    # ------------------------------------------------------------------
    # hard negatives
    # ------------------------------------------------------------------

    def corrupt(self, value: str, validator: Callable[[str], bool],
                max_tries: int = 100) -> str:
        """Keep the formatting, break the number, and confirm it now fails.

        70%: change one digit (a typo). 30%: swap two neighbouring digits
        (a transposition). We re-check every candidate because weak
        checksums (like the ACN's mod 10) can accidentally still pass.
        """
        positions = [i for i, ch in enumerate(value) if ch in "0123456789"]
        for _ in range(max_tries):
            chars = list(value)
            if self.rng.random() < 0.7 or len(positions) < 2:
                i = self.rng.choice(positions)
                chars[i] = self.rng.choice([d for d in "0123456789" if d != chars[i]])
            else:
                k = self.rng.randrange(len(positions) - 1)
                i, j = positions[k], positions[k + 1]
                chars[i], chars[j] = chars[j], chars[i]
            candidate = "".join(chars)
            if not validator(candidate):
                return candidate
        raise RuntimeError(f"could not corrupt {value!r} into an invalid value")

    # ------------------------------------------------------------------
    # label-based access (used by Level 2)
    # ------------------------------------------------------------------

    def generate(self, label: str, **kwargs) -> str:
        """Generate one valid value for a label from labels.py."""
        dispatch: dict[str, Callable[..., str]] = {
            "AU_TFN": self.tfn,
            "AU_ABN": self.abn,
            "AU_ACN": self.acn,
            "AU_MEDICARE": self.medicare,
            "AU_BSB": self.bsb,
            "AU_BANK_ACCOUNT": self.bank_account,
            "AU_PASSPORT": self.passport,
            "AU_DRIVER_LICENCE": self.driver_licence,
            "AU_PHONE": self.phone,
            "AU_ADDRESS": self.address,
            "PERSON": self.person_name,
            "EMAIL": self.email,
            "DATE_OF_BIRTH": self.date_of_birth,
        }
        if label not in dispatch:
            raise KeyError(f"unknown label {label!r}")
        return dispatch[label](**kwargs)

    def invalid(self, label: str, **kwargs) -> str:
        """A checksum-failing lookalike for AU_TFN/ABN/ACN/MEDICARE."""
        if label not in ("AU_TFN", "AU_ABN", "AU_ACN", "AU_MEDICARE"):
            raise ValueError("hard negatives only make sense for checksum labels")
        return self.corrupt(self.generate(label, **kwargs), v.VALIDATORS[label])
