"""Customer records from three systems (CRM, billing, support) describing the same people, messily.

Each person appears 1 to 5 times. A record may carry:
  - a nickname (Robert -> Bob), a typo, an initial, or no first name
  - a surname typo or a changed surname (marriage), or first and last name swapped
  - a missing date of birth, day and month swapped, or a mistyped year
  - a missing, different (new provider) or reformatted email
  - a missing, reformatted or different (new) phone number
  - an abbreviated street (Street -> St), a typo, or a whole new address (they moved)

The hard part is people who are *different* but look alike: very common names, spouses and children at
the same address, twins (same surname, address and birthday), and fathers and sons with the same name.
Every record keeps its true person id, so every decision can be graded.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

FIRST = {  # name: relative frequency, nicknames
    "James": (9, ["Jim", "Jimmy", "Jamie"]),
    "John": (9, ["Jack", "Johnny"]),
    "Robert": (8, ["Bob", "Rob", "Bobby"]),
    "Michael": (8, ["Mike", "Mick"]),
    "William": (7, ["Bill", "Will", "Billy"]),
    "David": (7, ["Dave"]),
    "Richard": (5, ["Rick", "Dick", "Rich"]),
    "Joseph": (5, ["Joe", "Joey"]),
    "Thomas": (5, ["Tom", "Tommy"]),
    "Charles": (4, ["Charlie", "Chuck"]),
    "Daniel": (4, ["Dan", "Danny"]),
    "Matthew": (4, ["Matt"]),
    "Anthony": (3, ["Tony"]),
    "Edward": (3, ["Ed", "Eddie", "Ted"]),
    "Christopher": (4, ["Chris"]),
    "Mary": (8, ["Molly", "Polly"]),
    "Patricia": (5, ["Pat", "Patty", "Trish"]),
    "Jennifer": (5, ["Jen", "Jenny"]),
    "Elizabeth": (6, ["Liz", "Beth", "Lizzie", "Betty"]),
    "Linda": (4, []),
    "Barbara": (4, ["Barb"]),
    "Susan": (4, ["Sue", "Susie"]),
    "Margaret": (4, ["Maggie", "Peggy", "Meg"]),
    "Katherine": (4, ["Kate", "Kathy", "Katie"]),
    "Rebecca": (3, ["Becky"]),
    "Deborah": (3, ["Debbie", "Deb"]),
    "Victoria": (2, ["Vicky", "Tori"]),
    "Alexander": (3, ["Alex", "Sandy"]),
    "Samantha": (2, ["Sam"]),
    "Olumide": (1, ["Mide"]),
    "Chidinma": (1, ["Chi"]),
    "Priya": (1, []),
    "Wei": (1, []),
    "Aisha": (1, []),
    "Mateo": (1, []),
    "Siobhan": (1, []),
    "Yusuf": (1, []),
}
LAST = {
    "Smith": 20,
    "Johnson": 14,
    "Williams": 12,
    "Brown": 11,
    "Jones": 11,
    "Miller": 8,
    "Davis": 8,
    "Wilson": 6,
    "Taylor": 6,
    "Clark": 5,
    "Lewis": 5,
    "Walker": 5,
    "Hall": 4,
    "Young": 4,
    "King": 4,
    "Wright": 4,
    "Scott": 3,
    "Green": 3,
    "Baker": 3,
    "Adams": 3,
    "Okafor": 2,
    "Adeyemi": 2,
    "Nwosu": 1,
    "Patel": 3,
    "Chen": 3,
    "Kowalski": 1,
    "Fitzgerald": 1,
    "Oyelaran": 1,
    "Haddad": 1,
    "Moreau": 1,
    "Lindqvist": 1,
    "Ferreira": 1,
    "Nakamura": 1,
}
STREETS = ["Oak", "Maple", "Cedar", "Elm", "Pine", "Station", "Church", "Mill", "Park", "Victoria", "High", "Kings"]
TYPES = {"Street": "St", "Road": "Rd", "Avenue": "Ave", "Lane": "Ln", "Drive": "Dr"}
CITIES = ["Leeds", "Bristol", "Manchester", "Glasgow", "Cardiff", "Norwich", "York", "Bath"]
DOMAINS = ["gmail.com", "outlook.com", "yahoo.co.uk", "icloud.com", "proton.me"]
SOURCES = ["crm", "billing", "support"]
NICK_TO_NAME = {n.lower(): f.lower() for f, (_, ns) in FIRST.items() for n in ns}


@dataclass(frozen=True)
class Record:
    id: int
    person: int
    source: str
    first: str
    last: str
    dob: str  # YYYY-MM-DD or ""
    email: str
    phone: str
    street: str
    city: str
    postcode: str
    kind: str  # how this person relates to a look-alike: "", "twin", "junior", "family"
    household: int  # people at one address share it: the look-alikes that must stay apart


@dataclass(frozen=True)
class Person:
    first: str
    last: str
    dob: tuple[int, int, int]
    email: str
    phone: str
    street: str
    city: str
    postcode: str
    kind: str
    household: int


def _weighted(rng: random.Random, table: dict[str, int] | dict[str, tuple[int, list[str]]]) -> str:
    names = list(table)
    weights = [v if isinstance(v, int) else v[0] for v in table.values()]
    return rng.choices(names, weights)[0]


def _address(rng: random.Random) -> tuple[str, str, str]:
    t = rng.choice(list(TYPES))
    outward = f"{rng.choice('ABLMNS')}{rng.choice('ABDEFS')}{rng.randint(1, 20)}"
    inward = f"{rng.randint(1, 9)}{rng.choice('ABDEFGHJ')}{rng.choice('LNPQRSTUW')}"
    return f"{rng.randint(1, 240)} {rng.choice(STREETS)} {t}", rng.choice(CITIES), f"{outward} {inward}"


def _phone(rng: random.Random) -> str:
    return "07" + "".join(rng.choice("0123456789") for _ in range(9))


_TAKEN: set[str] = set()


def _email(rng: random.Random, first: str, last: str) -> str:
    """An address nobody else has (mail providers don't hand out the same address twice)."""
    while True:
        pat = rng.choice(["{f}.{l}", "{f}{l}", "{fi}{l}", "{f}_{l}{n}", "{l}{n}"])
        local = pat.format(f=first.lower(), l=last.lower(), fi=first[0].lower(), n=rng.randint(1, 999))
        address = f"{local}@{rng.choice(DOMAINS)}"
        if address.replace(".", "").split("+")[0] not in _TAKEN:
            _TAKEN.add(address.replace(".", "").split("+")[0])
            return address


def people(n: int, rng: random.Random) -> list[Person]:
    out: list[Person] = []
    while len(out) < n:
        first, last = _weighted(rng, FIRST), _weighted(rng, LAST)
        dob = (rng.randint(1940, 2004), rng.randint(1, 12), rng.randint(1, 28))
        street, city, pc = _address(rng)
        home = len(out)
        p = Person(first, last, dob, _email(rng, first, last), _phone(rng), street, city, pc, "", home)
        out.append(p)
        r = rng.random()
        if r < 0.10:  # a family member at the same address
            f2 = _weighted(rng, FIRST)
            out.append(
                Person(
                    f2,
                    last,
                    (rng.randint(1940, 2004), rng.randint(1, 12), rng.randint(1, 28)),
                    _email(rng, f2, last),
                    _phone(rng),
                    street,
                    city,
                    pc,
                    "family",
                    home,
                )
            )
        elif r < 0.13:  # a twin: same surname, address and birthday
            f2 = _weighted(rng, FIRST)
            while f2 == first:
                f2 = _weighted(rng, FIRST)
            out.append(Person(f2, last, dob, _email(rng, f2, last), _phone(rng), street, city, pc, "twin", home))
        elif r < 0.16:  # a son with his father's name, at the same address
            out.append(
                Person(
                    first,
                    last,
                    (dob[0] + rng.randint(22, 35), rng.randint(1, 12), rng.randint(1, 28)),
                    _email(rng, first, last),
                    _phone(rng),
                    street,
                    city,
                    pc,
                    "junior",
                    home,
                )
            )
    return out[:n]


def _typo(rng: random.Random, s: str) -> str:
    if len(s) < 3:
        return s
    i = rng.randrange(1, len(s) - 1)
    op = rng.choice(["swap", "drop", "dup", "sub"])
    if op == "swap":
        return s[:i] + s[i + 1] + s[i] + s[i + 2 :]
    if op == "drop":
        return s[:i] + s[i + 1 :]
    if op == "dup":
        return s[:i] + s[i] + s[i:]
    return s[:i] + rng.choice("aeinorst") + s[i + 1 :]


def _record(
    rng: random.Random,
    rid: int,
    pid: int,
    p: Person,
    moved: tuple[str, str, str] | None,
    new_email: str | None,
    new_phone: str | None,
) -> Record:
    first, last = p.first, p.last
    r = rng.random()
    if r < 0.18 and FIRST[p.first][1]:
        first = rng.choice(FIRST[p.first][1])
    elif r < 0.28:
        first = _typo(rng, first)
    elif r < 0.33:
        first = first[0]
    elif r < 0.36:
        first = ""
    r = rng.random()
    if r < 0.08:
        last = _typo(rng, last)
    elif r < 0.10:
        last = _weighted(rng, LAST)  # changed surname
    if rng.random() < 0.02:
        first, last = last, first
    y, m, d = p.dob
    r = rng.random()
    dob = f"{y:04d}-{m:02d}-{d:02d}"
    if r < 0.2:
        dob = ""
    elif r < 0.25 and d <= 12:
        dob = f"{y:04d}-{d:02d}-{m:02d}"
    elif r < 0.28:
        dob = f"{y + rng.choice([-1, 1, 10, -10]):04d}-{m:02d}-{d:02d}"
    email = new_email if new_email and rng.random() < 0.5 else p.email
    r = rng.random()
    if r < 0.3:
        email = ""
    elif r < 0.4:
        local, dom = email.split("@")
        email = (
            f"{local.replace('.', '')}@{dom}".upper()
            if rng.random() < 0.5
            else f"{local}+{rng.choice(['shop', 'news'])}@{dom}"
        )
    phone = new_phone if new_phone and rng.random() < 0.5 else p.phone
    r = rng.random()
    if r < 0.3:
        phone = ""
    elif r < 0.6:
        phone = rng.choice(
            [f"+44 {phone[1:5]} {phone[5:]}", f"{phone[:5]} {phone[5:8]} {phone[8:]}", f"({phone[:5]}) {phone[5:]}"]
        )
    street, city, pc = moved if moved and rng.random() < 0.5 else (p.street, p.city, p.postcode)
    r = rng.random()
    if r < 0.3:
        for full, abbr in TYPES.items():
            street = street.replace(full, abbr)
    elif r < 0.38:
        street = _typo(rng, street)
    elif r < 0.45:
        street, pc = "", ""
    if rng.random() < 0.3:
        first, last, street, city = first.upper(), last.upper(), street.upper(), city.upper()
    return Record(rid, pid, rng.choice(SOURCES), first, last, dob, email, phone, street, city, pc, p.kind, p.household)


def generate(n_people: int = 6000, seed: int = 7) -> list[Record]:
    rng = random.Random(seed)
    _TAKEN.clear()
    ps = people(n_people, rng)
    out: list[Record] = []
    for pid, p in enumerate(ps):
        n = rng.choices([1, 2, 3, 4, 5], [25, 30, 25, 12, 8])[0]
        moved = _address(rng) if rng.random() < 0.15 else None
        new_email = _email(rng, p.first, p.last) if rng.random() < 0.2 else None
        new_phone = _phone(rng) if rng.random() < 0.15 else None
        for _ in range(n):
            out.append(_record(rng, len(out), pid, p, moved, new_email, new_phone))
    rng.shuffle(out)
    return [
        Record(
            i,
            *(
                getattr(r, f)
                for f in (
                    "person",
                    "source",
                    "first",
                    "last",
                    "dob",
                    "email",
                    "phone",
                    "street",
                    "city",
                    "postcode",
                    "kind",
                    "household",
                )
            ),
        )
        for i, r in enumerate(out)
    ]
