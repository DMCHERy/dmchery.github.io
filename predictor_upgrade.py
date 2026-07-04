import argparse
import json
import os
import re
import sys
from datetime import datetime
from html import unescape
from urllib.parse import quote_plus

import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# -----------------------------
# Original Manifold predictor
# -----------------------------

def fetch_markets():
    url = "https://api.manifold.markets/v0/markets"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def filter_binary_resolved(markets):
    return [
        m for m in markets
        if m['outcomeType'] == 'BINARY'
        and m.get('isResolved') is True
        and m.get('resolution') in ['YES', 'NO']
    ]


def get_gpt_opinion_summary(question):
    try:
        messages = [
            {"role": "system", "content": "You are a probability estimation expert."},
            {"role": "user", "content": f"Estimate the probability (between 0 and 1) that the answer to the question will be YES: {question}\nRespond ONLY with the number."}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3
        )
        reply = response.choices[0].message.content.strip()
        score = float(reply) if 0 <= float(reply) <= 1 else 0.5
        return {'yes_confidence': score}
    except Exception:
        return {'yes_confidence': 0.5}


def build_feature_set(markets):
    data = []
    for m in markets:
        try:
            gpt = get_gpt_opinion_summary(m['question'])
            row = {
                'id': m['id'],
                'question': m['question'],
                'questionLength': len(m['question']),
                'volume': m.get('volume', 0),
                'numTraders': m.get('uniqueBettorCount', 0),
                'timeOpen': (m['closeTime'] - m['createdTime']) / (1000 * 60 * 60 * 24),
                'gpt_pos_confidence': gpt['yes_confidence'],
                'label': 1 if m['resolution'] == 'YES' else 0
            }
            data.append(row)
        except Exception:
            continue
    return pd.DataFrame(data)


def train_and_evaluate(df):
    X = df.drop(columns=["id", "question", "label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")

    importance = sorted(zip(X.columns, clf.feature_importances_), key=lambda x: -x[1])
    print("\nFeature Importance:")
    for feat, score in importance:
        print(f"{feat}: {score:.4f}")

    return clf


def fetch_unresolved_binary():
    url = "https://api.manifold.markets/v0/markets"
    response = requests.get(url)
    response.raise_for_status()
    return [
        m for m in response.json()
        if m['outcomeType'] == 'BINARY'
        and not m.get('isResolved', False)
    ]


def build_unresolved_features(markets):
    data = []
    ids = []
    questions = []
    links = []
    dates = []
    for m in markets:
        try:
            gpt = get_gpt_opinion_summary(m['question'])
            row = {
                'questionLength': len(m['question']),
                'volume': m.get('volume', 0),
                'numTraders': m.get('uniqueBettorCount', 0),
                'timeOpen': (m['closeTime'] - m['createdTime']) / (1000 * 60 * 60 * 24),
                'gpt_pos_confidence': gpt['yes_confidence']
            }
            data.append(row)
            ids.append(m['id'])
            questions.append(m['question'])
            links.append(f"https://manifold.markets/{m['creatorUsername']}/{m['slug']}")
            dates.append(datetime.utcnow().strftime('%Y-%m-%d'))
        except Exception:
            continue

    df = pd.DataFrame(data)
    df['id'] = ids
    df['question'] = questions
    df['link'] = links
    df['date'] = dates
    return df


def predict_unresolved(model):
    print("\nFetching unresolved binary markets...")
    unresolved = fetch_unresolved_binary()
    if not unresolved:
        print("No unresolved binary markets found.")
        return

    df_features = build_unresolved_features(unresolved)
    X = df_features.drop(columns=["id", "question", "link", "date"])

    print("Predicting...")
    df_features['predicted_prob_yes'] = model.predict_proba(X)[:, 1]

    top = df_features.sort_values(by='predicted_prob_yes', ascending=False).head(10)
    print("\nTop Predictions:")
    print(top[['question', 'predicted_prob_yes']])

    df_features.to_csv("manifold_predictions.csv", index=False)
    print("Predictions saved to 'manifold_predictions.csv'")

    predictions_json = df_features[['question', 'predicted_prob_yes', 'link', 'date']].copy()
    predictions_json.rename(columns={"predicted_prob_yes": "probability"}, inplace=True)
    predictions_json.to_json("predictions.json", orient="records", indent=2)
    print("Predictions also saved to 'predictions.json'")


def run_manifold_predictor():
    try:
        print("Fetching markets...")
        markets = fetch_markets()
        resolved = filter_binary_resolved(markets)

        print("\nBuilding features...")
        df = build_feature_set(resolved)

        if df.empty:
            print("No resolved binary markets available.")
        else:
            print(f"Training on {len(df)} resolved markets...")
            clf = train_and_evaluate(df)
            predict_unresolved(clf)

    except Exception as e:
        print(f"Error: {e}")


# -----------------------------
# PIN-gated NY job finder
# -----------------------------

JOB_PIN = os.getenv("JOB_PIN", "1469")
NY_JOB_LOCATION = os.getenv("NY_JOB_LOCATION", "New York, NY")
MAX_JOBS_PER_SOURCE = int(os.getenv("MAX_JOBS_PER_SOURCE", "50"))
MAX_FRONTEND_JOBS = int(os.getenv("MAX_FRONTEND_JOBS", "75"))

# These terms are limited to legitimate job fit signals: language, public job titles,
# community-facing workplaces, and entry-level/pay signals. Do not use this to target,
# exclude, or manipulate people based on protected traits.
JOB_SEARCH_TERMS = [
    "no experience",
    "entry level",
    "paid training",
    "will train",
    "front desk",
    "receptionist",
    "concierge",
    "office assistant",
    "customer service",
    "leasing assistant",
    "sales associate",
    "mover helper",
    "moving helper",
    "warehouse helper",
    "driver helper",
    "carpenter helper",
    "construction helper",
    "maintenance helper",
    "Russian speaking",
    "bilingual Russian",
    "Russian English",
    "Uzbek speaking",
    "Kazakh speaking",
    "Central Asian",
    "Bukharian",
    "kosher",
    "JCC",
    "Jewish community center",
    "caregiver Russian",
    "home care Russian",
]

ENTRY_LEVEL_TERMS = [
    "no experience", "entry level", "entry-level", "paid training", "will train",
    "training provided", "no degree", "no previous", "beginner", "trainee",
]

HIGH_PAY_TERMS = [
    "bonus", "sign-on", "sign on", "commission", "tips", "overtime", "weekly pay",
    "daily pay", "high paying", "premium", "differential",
]

LANGUAGE_COMMUNITY_TERMS = [
    "russian", "bilingual", "uzbek", "kazakh", "central asian", "bukharian",
    "hebrew", "yiddish", "kosher", "jcc", "jewish community center",
]

HEALTHCARE_TERMS = [
    "cna", "certified nursing assistant", "nursing assistant", "patient care", "caregiver",
    "home care", "hha", "companion", "aide", "senior care",
]

FRONT_DESK_TERMS = [
    "front desk", "receptionist", "concierge", "office assistant", "admin",
    "administrative", "customer service", "leasing assistant", "clerk",
]

HANDS_ON_TERMS = [
    "mover", "moving", "warehouse", "helper", "driver helper", "carpenter",
    "construction", "maintenance", "installer", "laborer", "porter",
]


def clean_html(raw_text):
    if not raw_text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(raw_text))
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def contains_any(text, terms):
    text_l = (text or "").lower()
    return any(term.lower() in text_l for term in terms)


def money_to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace("$", "").replace(",", ""))
    except ValueError:
        return None


def extract_salary_from_text(text):
    text = text or ""
    # Examples: $25 an hour, $20 - $30/hour, $55,000 a year
    hourly = re.findall(r"\$\s*([0-9]{2,3}(?:\.[0-9]{1,2})?)\s*(?:-|to)?\s*\$?\s*([0-9]{2,3}(?:\.[0-9]{1,2})?)?\s*(?:/|per\s*)?(?:hr|hour)", text, flags=re.I)
    if hourly:
        vals = []
        for a, b in hourly:
            vals.append(float(a))
            if b:
                vals.append(float(b))
        return min(vals), max(vals), f"${min(vals):.2f} - ${max(vals):.2f}/hour" if len(vals) > 1 else f"${vals[0]:.2f}/hour"

    yearly = re.findall(r"\$\s*([0-9]{2,3}(?:,[0-9]{3})+|[0-9]{5,6})\s*(?:-|to)?\s*\$?\s*([0-9]{2,3}(?:,[0-9]{3})+|[0-9]{5,6})?\s*(?:/|per\s*)?(?:yr|year|annually|annual)", text, flags=re.I)
    if yearly:
        vals = []
        for a, b in yearly:
            vals.append(float(a.replace(",", "")))
            if b:
                vals.append(float(b.replace(",", "")))
        return min(vals), max(vals), f"${min(vals):,.0f} - ${max(vals):,.0f}/year" if len(vals) > 1 else f"${vals[0]:,.0f}/year"

    return None, None, ""


def normalize_salary(job):
    salary_min = money_to_float(job.get("salary_min"))
    salary_max = money_to_float(job.get("salary_max"))
    salary_text = job.get("salary", "") or ""

    if salary_min or salary_max:
        lo = salary_min or salary_max
        hi = salary_max or salary_min
        if hi and hi > 1000:
            salary_text = f"${lo:,.0f} - ${hi:,.0f}/year"
        else:
            salary_text = f"${lo:.2f} - ${hi:.2f}/hour"
        return lo, hi, salary_text

    text = f"{job.get('title', '')} {job.get('description', '')} {job.get('salary', '')}"
    return extract_salary_from_text(text)


def category_and_presets(title, description):
    text = f"{title} {description}".lower()
    if contains_any(text, HEALTHCARE_TERMS):
        return "Healthcare / CNA / Caregiver", "Resume 1 - Healthcare/CNA", "Email 1 - Direct application"
    if contains_any(text, FRONT_DESK_TERMS):
        return "Front Desk / Customer Service", "Resume 2 - Front Desk/Admin", "Email 1 - Direct application"
    if contains_any(text, HANDS_ON_TERMS):
        return "Hands-on / Helper / Moving", "Resume 3 - Helper/Moving/Warehouse", "Email 1 - Direct application"
    return "Entry-level / General", "Resume 2 - Front Desk/Admin", "Email 2 - Relocation inquiry"


def score_job(job):
    title = job.get("title", "")
    description = job.get("description", "")
    text = f"{title} {description} {job.get('company', '')} {job.get('location', '')}".lower()
    score = 0
    reasons = []

    salary_min, salary_max, salary_text = normalize_salary(job)
    job["salary_min"] = salary_min
    job["salary_max"] = salary_max
    job["salary"] = job.get("salary") or salary_text

    if contains_any(text, ENTRY_LEVEL_TERMS):
        score += 25
        reasons.append("entry-level/no-experience signal")

    if contains_any(text, HIGH_PAY_TERMS):
        score += 10
        reasons.append("pay/bonus signal")

    if contains_any(text, LANGUAGE_COMMUNITY_TERMS):
        score += 18
        reasons.append("language/community-facing fit")

    if contains_any(text, HEALTHCARE_TERMS):
        score += 18
        reasons.append("healthcare/CNA fit")

    if contains_any(text, FRONT_DESK_TERMS):
        score += 16
        reasons.append("front desk/customer-facing fit")

    if contains_any(text, HANDS_ON_TERMS):
        score += 16
        reasons.append("hands-on/helper fit")

    # Pay scoring: hourly or yearly. We do not invent pay when it is missing.
    pay_ref = salary_max or salary_min
    if pay_ref:
        if pay_ref <= 100:  # likely hourly
            if pay_ref >= 30:
                score += 25
                reasons.append("strong hourly pay")
            elif pay_ref >= 25:
                score += 18
                reasons.append("good hourly pay")
            elif pay_ref >= 20:
                score += 10
                reasons.append("acceptable hourly pay")
        else:  # likely annual
            if pay_ref >= 70000:
                score += 25
                reasons.append("strong annual pay")
            elif pay_ref >= 55000:
                score += 15
                reasons.append("good annual pay")

    if "new york" in text or "brooklyn" in text or "queens" in text or "manhattan" in text or "bronx" in text or "staten island" in text:
        score += 10
        reasons.append("NY location")

    # Reduce score for jobs that strongly imply advanced experience/credentials.
    advanced_terms = ["5+ years", "7+ years", "bachelor", "master", "rn", "registered nurse", "license required", "cdl required", "journeyman"]
    if contains_any(text, advanced_terms):
        score -= 20
        reasons.append("may require advanced credentials/experience")

    return max(score, 0), reasons


def make_email_draft(job):
    title = job.get("title", "the position")
    company = job.get("company") or "your team"
    category = job.get("category", "")
    subject = f"Application for {title}"

    if "Healthcare" in category:
        body = f"""Dear Hiring Manager,

I hope you are doing well.

I am reaching out to express my interest in the {title} position at {company}. I have hands-on healthcare/patient care experience and I am interested in continuing my career in New York.

My experience includes assisting with daily care, transfers, ambulation, feeding, safety monitoring, and working respectfully with residents, patients, families, and the healthcare team.

I have attached my resume for your review. I would appreciate the opportunity to discuss my availability and how I can be a good fit for your team.

Best regards,
Gulnara Burbayeva
[Phone Number]
[Email]
"""
    elif "Hands-on" in category:
        body = f"""Dear Hiring Manager,

I hope you are doing well.

I am interested in the {title} position at {company}. I am reliable, hands-on, physically capable, and ready to learn. I am looking for an opportunity in New York where strong work ethic, responsibility, and consistency matter.

I have attached my resume for your review and would be glad to discuss my availability.

Best regards,
Gulnara Burbayeva
[Phone Number]
[Email]
"""
    else:
        body = f"""Dear Hiring Manager,

I hope you are doing well.

I am writing to express my interest in the {title} position at {company}. I am planning to work in New York and I am especially interested in roles where reliability, communication, customer service, and bilingual/community-facing experience are valuable.

I have attached my resume for your review. I would appreciate the opportunity to be considered for this position and would be glad to discuss my availability.

Best regards,
Gulnara Burbayeva
[Phone Number]
[Email]
"""

    return subject, body


def normalize_job(raw):
    title = clean_html(raw.get("title"))
    company = clean_html(raw.get("company"))
    description = clean_html(raw.get("description"))
    location = clean_html(raw.get("location")) or NY_JOB_LOCATION
    url = raw.get("url") or raw.get("redirect_url") or ""

    job = {
        "source": raw.get("source", ""),
        "title": title,
        "company": company,
        "location": location,
        "salary": raw.get("salary", "") or "",
        "salary_min": raw.get("salary_min"),
        "salary_max": raw.get("salary_max"),
        "description": description,
        "url": url,
        "date_found": datetime.utcnow().strftime("%Y-%m-%d"),
    }

    score, reasons = score_job(job)
    category, resume_preset, email_preset = category_and_presets(title, description)
    subject, body = make_email_draft({**job, "category": category})

    job.update({
        "fit_score": score,
        "fit_reasons": reasons,
        "category": category,
        "resume_preset": resume_preset,
        "email_preset": email_preset,
        "draft_subject": subject,
        "draft_email": body,
    })
    return job


def fetch_adzuna_jobs():
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        print("Adzuna keys not found. Skipping Adzuna. Add ADZUNA_APP_ID and ADZUNA_APP_KEY to .env for stronger search.")
        return []

    jobs = []
    seen = set()
    for term in JOB_SEARCH_TERMS:
        url = "https://api.adzuna.com/v1/api/jobs/us/search/1"
        params = {
            "app_id": app_id,
            "app_key": app_key,
            "what": term,
            "where": NY_JOB_LOCATION,
            "results_per_page": min(MAX_JOBS_PER_SOURCE, 50),
            "sort_by": "date",
        }
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            for item in response.json().get("results", []):
                key = item.get("id") or item.get("redirect_url")
                if key in seen:
                    continue
                seen.add(key)
                company = item.get("company", {}) or {}
                location = item.get("location", {}) or {}
                jobs.append(normalize_job({
                    "source": "Adzuna",
                    "title": item.get("title", ""),
                    "company": company.get("display_name", ""),
                    "location": location.get("display_name", NY_JOB_LOCATION),
                    "description": item.get("description", ""),
                    "url": item.get("redirect_url", ""),
                    "salary_min": item.get("salary_min"),
                    "salary_max": item.get("salary_max"),
                }))
        except Exception as e:
            print(f"Adzuna error for '{term}': {e}")
    return jobs


def fetch_themuse_jobs():
    jobs = []
    seen = set()
    # The Muse is a no-key fallback. It may not return salary, so those jobs are scored mainly by title/description fit.
    for page in range(1, 4):
        url = "https://www.themuse.com/api/public/jobs"
        params = {
            "page": page,
            "location": "New York City, NY",
            "descending": "true",
        }
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            for item in data.get("results", []):
                levels = " ".join([level.get("name", "") for level in item.get("levels", [])])
                contents = clean_html(item.get("contents", ""))
                text_for_filter = f"{item.get('name', '')} {contents} {levels}".lower()

                # Keep only jobs with some evidence of matching the user's requested buckets.
                if not (
                    contains_any(text_for_filter, ENTRY_LEVEL_TERMS)
                    or contains_any(text_for_filter, FRONT_DESK_TERMS)
                    or contains_any(text_for_filter, HANDS_ON_TERMS)
                    or contains_any(text_for_filter, HEALTHCARE_TERMS)
                    or contains_any(text_for_filter, LANGUAGE_COMMUNITY_TERMS)
                ):
                    continue

                key = item.get("id") or item.get("refs", {}).get("landing_page")
                if key in seen:
                    continue
                seen.add(key)

                company = item.get("company", {}) or {}
                locations = item.get("locations", []) or []
                location = ", ".join([loc.get("name", "") for loc in locations]) or NY_JOB_LOCATION
                jobs.append(normalize_job({
                    "source": "The Muse",
                    "title": item.get("name", ""),
                    "company": company.get("name", ""),
                    "location": location,
                    "description": contents,
                    "url": item.get("refs", {}).get("landing_page", ""),
                }))
        except Exception as e:
            print(f"The Muse error on page {page}: {e}")
    return jobs


def dedupe_jobs(jobs):
    deduped = {}
    for job in jobs:
        key = (
            re.sub(r"\W+", "", job.get("title", "").lower()),
            re.sub(r"\W+", "", job.get("company", "").lower()),
            re.sub(r"\W+", "", job.get("location", "").lower())[:40],
        )
        if key not in deduped or job.get("fit_score", 0) > deduped[key].get("fit_score", 0):
            deduped[key] = job
    return list(deduped.values())


def run_ny_job_finder(require_pin=True):
    if require_pin:
        if not sys.stdin.isatty():
            print("PIN prompt requested in a non-interactive environment. Use --no-pin for GitHub Actions/CI.")
            return []
        pin = input("Enter NY jobs PIN: ").strip()
        if pin != JOB_PIN:
            print("Incorrect PIN. NY job finder did not run.")
            return []

    print("Searching NY job leads...")
    jobs = []
    jobs.extend(fetch_adzuna_jobs())
    jobs.extend(fetch_themuse_jobs())

    jobs = dedupe_jobs(jobs)
    jobs.sort(key=lambda x: x.get("fit_score", 0), reverse=True)
    jobs = jobs[:MAX_FRONTEND_JOBS]

    if not jobs:
        print("No NY job leads found. Try adding Adzuna keys in .env or expanding JOB_SEARCH_TERMS.")
    else:
        print(f"Found {len(jobs)} job leads.")

    df = pd.DataFrame(jobs)
    df.to_csv("ny_job_leads.csv", index=False)
    print("Saved job leads to 'ny_job_leads.csv'")

    frontend_jobs = []
    drafts = []
    for job in jobs:
        frontend_jobs.append({
            "title": job.get("title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "salary": job.get("salary"),
            "source": job.get("source"),
            "category": job.get("category"),
            "fit_score": job.get("fit_score"),
            "fit_reasons": job.get("fit_reasons"),
            "resume_preset": job.get("resume_preset"),
            "email_preset": job.get("email_preset"),
            "description": job.get("description"),
            "url": job.get("url"),
            "date_found": job.get("date_found"),
            "draft_subject": job.get("draft_subject"),
            "draft_email": job.get("draft_email"),
        })
        drafts.append({
            "job_title": job.get("title"),
            "company": job.get("company"),
            "resume_preset": job.get("resume_preset"),
            "email_preset": job.get("email_preset"),
            "subject": job.get("draft_subject"),
            "body": job.get("draft_email"),
            "url": job.get("url"),
        })

    with open("ny_job_leads.json", "w", encoding="utf-8") as f:
        json.dump(frontend_jobs, f, indent=2, ensure_ascii=False)
    print("Saved frontend job data to 'ny_job_leads.json'")

    with open("job_outreach_drafts.json", "w", encoding="utf-8") as f:
        json.dump(drafts, f, indent=2, ensure_ascii=False)
    print("Saved email drafts to 'job_outreach_drafts.json'")

    with open("job_outreach_drafts.txt", "w", encoding="utf-8") as f:
        for draft in drafts:
            f.write("=" * 80 + "\n")
            f.write(f"JOB: {draft['job_title']} — {draft['company']}\n")
            f.write(f"RESUME PRESET: {draft['resume_preset']}\n")
            f.write(f"EMAIL PRESET: {draft['email_preset']}\n")
            f.write(f"SUBJECT: {draft['subject']}\n\n")
            f.write(draft["body"])
            f.write("\n")
            f.write(f"LINK: {draft['url']}\n\n")
    print("Saved readable drafts to 'job_outreach_drafts.txt'")

    print("\nTop job leads:")
    for job in jobs[:10]:
        salary = job.get("salary") or "salary not listed"
        print(f"- [{job.get('fit_score')}] {job.get('title')} — {job.get('company')} — {salary}")

    return jobs


def main():
    parser = argparse.ArgumentParser(description="Manifold predictor + PIN-gated NY job lead generator")
    parser.add_argument("--jobs", action="store_true", help="Run only the PIN-gated NY job finder")
    parser.add_argument("--all", action="store_true", help="Run Manifold predictor, then the PIN-gated NY job finder")
    parser.add_argument("--no-pin", action="store_true", help="Developer mode: run jobs without PIN prompt")
    args = parser.parse_args()

    if args.jobs:
        run_ny_job_finder(require_pin=not args.no_pin)
    elif args.all:
        run_manifold_predictor()
        run_ny_job_finder(require_pin=not args.no_pin)
    else:
        run_manifold_predictor()


if __name__ == "__main__":
    main()
