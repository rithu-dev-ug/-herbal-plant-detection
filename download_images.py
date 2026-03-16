import requests
import os
import time
from PIL import Image
from io import BytesIO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SAVE_DIR = "dataset/centella"
TARGET_IMAGES = 80
START_PAGE = 6
PER_PAGE = 200
MIN_WIDTH = 600
MIN_HEIGHT = 600

os.makedirs(SAVE_DIR, exist_ok=True)


def get_taxon_id(name):
    r = requests.get(
        "https://api.inaturalist.org/v1/taxa",
        params={"q": name}
    ).json()
    return r["results"][0]["id"]


def is_good_leaf_candidate(img_bytes):
    try:
        img = Image.open(BytesIO(img_bytes))
        w, h = img.size

        if w < MIN_WIDTH or h < MIN_HEIGHT:
            return False

        ratio = max(w, h) / min(w, h)

        # extremely wide images usually show whole plants
        if ratio > 2.5:
            return False

        return True
    except:
        return False


print("Fetching taxon id for Centella asiatica...")

taxon_id = get_taxon_id("Centella asiatica")

count = 0
page = START_PAGE

print("Downloading additional centella leaf images...")

while count < TARGET_IMAGES:

    params = {
        "taxon_id": taxon_id,
        "photos": "true",
        "quality_grade": "research",
        "per_page": PER_PAGE,
        "page": page
    }

    data = requests.get(
        "https://api.inaturalist.org/v1/observations",
        params=params
    ).json()

    results = data["results"]

    if not results:
        break

    for obs in results:

        if count >= TARGET_IMAGES:
            break

        photos = obs.get("photos")

        if not photos:
            continue

        img_url = photos[0]["url"].replace("square", "large")

        try:
            img_bytes = requests.get(img_url, timeout=10).content
        except:
            continue

        if not is_good_leaf_candidate(img_bytes):
            continue

        filename = os.path.join(
            SAVE_DIR,
            f"centella_leaf_{count}.jpg"
        )

        with open(filename, "wb") as f:
            f.write(img_bytes)

        count += 1
        print("Downloaded", count)

    page += 1
    time.sleep(1)

print("Finished downloading centella leaf dataset.")
