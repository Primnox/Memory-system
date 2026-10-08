"""Random people for the chat-format data: no hand-picked list, so the model meets
lives and changes nobody planned for.   python sampler.py 5 --seed 7  (prints samples)

Each person is drawn from wide pools (where, age, work, household, health, money),
gets random life areas to carry the updates, random event types from a large
library, two changes the writer must INVENT itself, random writing-style traits and
random chat purposes. Seeded: the same seed gives the same people.
"""
from __future__ import annotations

import random
import sys
from datetime import date, timedelta

REGIONS = [
    "rural Ireland", "a town in northern Sweden", "Marseille", "a village in the Basque Country", "Kraków",
    "Bucharest", "Sarajevo", "Tbilisi", "Yerevan", "Izmir", "Beirut", "Amman", "Tehran", "Muscat", "Doha",
    "Karachi", "Lahore", "Kathmandu", "Dhaka", "Colombo", "Chennai", "Kochi", "Ahmedabad", "Jaipur",
    "Chiang Mai", "Hanoi", "Da Nang", "Phnom Penh", "Yangon", "Kuala Lumpur", "Surabaya", "Davao", "Taipei",
    "Busan", "Fukuoka", "Hokkaido farm country", "Ulaanbaatar", "Almaty", "Bishkek", "Novosibirsk",
    "Addis Ababa", "Kigali", "Dar es Salaam", "Lusaka", "Harare", "Gaborone", "Windhoek", "Durban",
    "Dakar", "Abidjan", "Kumasi", "Ibadan", "Douala", "Kinshasa", "Tunis", "Casablanca", "Alexandria",
    "Reykjavik", "the Faroe Islands", "Halifax", "rural Saskatchewan", "Anchorage", "New Orleans", "El Paso",
    "rural Kentucky", "Minneapolis", "Oaxaca", "Guatemala City", "San José (Costa Rica)", "Havana",
    "Kingston (Jamaica)", "Port of Spain", "Bogotá", "Quito", "Cusco", "La Paz", "Asunción", "Salvador (Bahia)",
    "Porto Alegre", "Mendoza", "Valparaíso", "Fiji", "Samoa", "Darwin", "Hobart", "Dunedin",
]
OCCUPATIONS = [
    "bus driver", "air-traffic controller", "midwife", "night-shift security guard", "fish-market auctioneer",
    "tattoo artist", "court interpreter", "dental hygienist", "oil-rig worker", "beekeeper", "piano tuner",
    "school bus mechanic", "wedding photographer", "hospice nurse", "street-food vendor", "insurance adjuster",
    "PhD student in marine biology", "retired postman", "unemployed and job-hunting", "stay-at-home parent",
    "carer for a disabled sibling", "long-haul pilot", "construction foreman", "pharmacy technician",
    "sign-language interpreter", "librarian", "football referee", "ski instructor", "taxi dispatcher",
    "data-entry clerk", "rice farmer", "tea-plantation manager", "ferry captain", "kindergarten aide",
    "hotel night auditor", "parliamentary researcher", "podcast producer", "mosque caretaker", "parish priest",
    "brewer", "veterinary nurse", "prison officer", "real-estate agent", "freelance translator", "welder",
    "museum guard", "radio DJ", "chemistry teacher", "call-centre agent", "warehouse picker", "tailor",
    "Uber Eats courier", "academic librarian", "flight attendant", "dairy-goat farmer", "IT help-desk tech",
    "retired engineer", "pastry chef", "hospital porter", "ballet teacher", "forest ranger", "notary",
    "mobile-phone repair shop owner", "nail technician", "marine mechanic", "school principal",
    "home-care worker", "sound engineer", "crossing guard", "tour-bus guide", "esports coach",
]
HOUSEHOLDS = [
    "lives alone", "married, no children", "single parent of one", "single parent of three",
    "lives with elderly parents", "shares a flat with two friends", "newly married", "divorced, kids every other week",
    "widowed", "long-distance relationship", "lives with partner and partner's child", "multigenerational household",
    "lives with a sibling", "polyamorous household of three adults", "foster parent", "empty-nester couple",
]
HEALTH = ["healthy", "type 1 diabetes", "asthma", "chronic back pain", "recovering from cancer", "ADHD",
          "a new pregnancy", "high blood pressure", "long COVID", "a hearing aid user", "in recovery from addiction",
          "migraines", "coeliac disease", "a knee replacement coming up", "depression, in therapy", "no known issues"]
MONEY = ["comfortable", "paycheck to paycheck", "paying off student debt", "saving for a house", "recently inherited money",
         "in debt after a failed business", "on a pension", "sends money home every month", "irregular gig income"]
AREAS = [
    "home and neighbourhood", "partner and relationship", "children and their schools", "parents and siblings",
    "job, employer and role", "manager and colleagues", "work hours and shifts", "pay and side income",
    "commute and vehicles", "phone, laptop and subscriptions", "bank, savings and debt", "rent or mortgage",
    "health conditions and doctors", "medications and doses", "allergies and diet", "exercise and sport",
    "hobbies and classes", "pets", "friends", "languages and education", "travel and visas", "faith and community",
    "volunteering", "routines and schedules", "important dates", "online accounts and usernames",
    "legal matters", "home appliances and repairs", "clubs and memberships", "favourite places and shops",
]
EVENTS = [
    "a move to a new home", "a move back to a previous home", "a new job", "a promotion with the same employer",
    "the employer is acquired and renamed (same job)", "a layoff", "a career change after retraining",
    "a new manager", "shift pattern changes twice", "a raise", "a pay cut", "a new phone, then going back to the old one",
    "a car sold and replaced", "a bike stolen", "a new pet", "a pet dies", "a pet is rehomed", "a break-up",
    "getting back together with an ex", "an engagement", "a wedding", "a divorce", "a baby", "a child changes schools",
    "a child moves out", "an adult child moves back home", "a parent moves in", "a parent dies", "a sibling emigrates",
    "a new diagnosis", "a medication started, stopped, then restarted", "a dose changes", "a diet becomes stricter",
    "a diet is abandoned", "a new allergy", "an injury that ends a sport", "a new sport", "a hobby dropped and resumed",
    "a class schedule moves", "a correction of something said earlier", "a name change", "a phone number change",
    "an email address change", "a new bank", "a loan paid off", "a new debt", "a citizenship or visa change",
    "a religious conversion or leaving a faith", "a new volunteer role", "a recurring appointment moves day",
    "a street or town is renamed (the person did not move)", "a friend moves to another city",
    "a partner changes jobs", "quitting smoking, relapsing, quitting again", "a subscription cancelled",
    "a club membership lapses", "a house sale falls through", "a renovation forces a temporary stay elsewhere",
    "going part-time", "retirement", "un-retiring", "a lottery or inheritance windfall",
]
STYLES = [
    "lots of typos and no capital letters", "voice dictation: long run-on sentences", "very terse, few words",
    "chatty, tells stories", "formal and polite", "second-language English with small grammar slips",
    "mixes in words from their own language", "uses emoji", "writes in bullet points", "sarcastic humour",
    "anxious, over-explains", "uses lots of abbreviations (tmrw, pls, bc)", "corrects themselves mid-message",
]
PURPOSES = [
    "planning a trip", "a recipe for guests", "an email to a landlord", "a workout plan", "a monthly budget",
    "a gift idea", "a child's homework", "a phone or laptop setting", "a cover letter", "comparing two products",
    "a complaint to a company", "a speech for a wedding", "a medical appointment checklist", "a garden plan",
    "learning a language", "a car repair quote", "a visa application form", "a party plan", "a resignation letter",
    "a sleep routine", "a reading list", "fixing a leaking tap", "translating a letter", "a pet's diet",
    "a dispute with a neighbour", "a tax question", "meal prep for shifts", "a school form", "a bus timetable",
]


YOUNG_NO = {"empty-nester couple", "widowed", "foster parent", "single parent of three", "divorced, kids every other week"}
SENIOR_JOBS_NO = ("PhD student", "kindergarten aide", "esports coach", "Uber Eats courier", "warehouse picker")


def article(phrase: str) -> str:
    if phrase.startswith(("unemployed", "stay-at-home", "carer")):
        return "someone " + phrase if phrase.startswith("unemployed") else "a " + phrase
    return ("an " if phrase[0].lower() in "aeiou" else "a ") + phrase


def person(rng: random.Random, n: int) -> dict:
    today = date(2026, 3, 1) + timedelta(days=rng.randrange(0, 218))
    age = rng.randint(18, 90)
    households = [h for h in HOUSEHOLDS if not (age < 25 and h in YOUNG_NO)]
    jobs = [j for j in OCCUPATIONS if not (age < 25 and j.startswith(("retired", "school principal", "notary")))
            and not (age > 65 and j.startswith(SENIOR_JOBS_NO))]
    job = rng.choice(jobs)
    if age > 72 and not job.startswith(("retired", "unemployed", "stay-at-home", "carer")):
        job = "retired " + job
    return {
        "who": (f"{article(job)} in {rng.choice(REGIONS)}, {age}, "
                f"{rng.choice(households)}; health: {rng.choice(HEALTH)}; money: {rng.choice(MONEY)}"),
        "areas": rng.sample(AREAS, 4),
        "events": rng.sample(EVENTS, 6),
        "style": rng.sample(STYLES, 2),
        "purposes": rng.sample(PURPOSES, 6),
        "today": today.isoformat(),
        "n": n,
    }


def assignment(p: dict) -> str:
    return (f"Person: {p['who']}.\n"
            f"The life areas that carry most of the updates: {', '.join(p['areas'])}.\n"
            f"Among the life changes, include these: {'; '.join(p['events'])}.\n"
            "ALSO INVENT two life changes of your own that are real but unusual, that none of the lists in this "
            "spec mention, and label them honestly.\n"
            f"Writing style: {p['style'][0]}; {p['style'][1]}.\n"
            f"Some of the chats are about: {', '.join(p['purposes'])} (and others of your choosing).\n"
            f"`today` = {p['today']}.\n"
            "These details were drawn at random and may clash. Keep the person believable: reinterpret any item "
            "that cannot fit as written (a 'child' can be a stepchild, a 'manager' a volunteer coordinator) or "
            "swap it for a similar change, and keep the rest. Odd but real lives are welcome; impossible ones are not.")


def sample(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [person(rng, i) for i in range(n)]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1
    for p in sample(n, seed):
        print(assignment(p), "\n")
