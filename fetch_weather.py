"""
fetch_weather.py
================
Fetches historical weather for every IPL venue on every match date
using Open-Meteo's free archive API (no key, no sign-up required).

Output: data/weather.csv
Columns: match_id, date, venue, temp_max_c, temp_min_c, humidity_eve_pct,
         precipitation_mm, cloud_cover_eve_pct, wind_speed_max_kmh,
         dew_point_eve_c

Changes in this version
-----------------------
FIX 1 — Added all 26 missing venues with verified GPS coordinates:
         17 India: Brabourne, DY Patil, Barabati, VCA Nagpur, Green Park,
                   Nehru Coimbatore, ACA-VDCA Vizag, Sahara/MCA Pune,
                   Shaheed Raipur, Saurashtra Rajkot, Ekana Lucknow,
                   Barsapara Guwahati, Mullanpur/New Chandigarh,
                   Sardar Patel/Motera, M.Chinnaswamy
         8  South Africa: Newlands, St George's, Kingsmead, SuperSport,
                          Buffalo Park, Wanderers, Diamond Oval, Outsurance
         1  UAE: Zayed/Abu Dhabi

FIX 2 — Normalised matching: strips punctuation and extra city suffixes
         before comparing, so 'M.Chinnaswamy' matches 'M Chinnaswamy',
         'Zayed Cricket Stadium, Abu Dhabi' matches 'Zayed Cricket Stadium'.

FIX 3 — STADIUM_ALIASES dict: maps renamed/alternate names to canonical key.
         Sardar Patel → Narendra Modi, Subrata Roy Sahara → MCA Pune, etc.
"""

import re
import pandas as pd
import numpy as np
import requests
import time

# ── Venue coordinates ─────────────────────────────────────────────────────────
VENUE_COORDS = {
    # ── Mumbai ────────────────────────────────────────────────────────────────
    "Wankhede Stadium"                              : (19.0445, 72.8258),
    "Brabourne Stadium"                             : (18.9330, 72.8296),
    "Dr DY Patil Sports Academy"                    : (19.1130, 72.9010),
    "Mumbai"                                        : (19.0445, 72.8258),

    # ── Bangalore ─────────────────────────────────────────────────────────────
    "M Chinnaswamy Stadium"                         : (12.9792, 77.5996),
    "Bangalore"                                     : (12.9792, 77.5996),

    # ── Kolkata ───────────────────────────────────────────────────────────────
    "Eden Gardens"                                  : (22.5645, 88.3433),
    "Kolkata"                                       : (22.5645, 88.3433),

    # ── Chennai ───────────────────────────────────────────────────────────────
    "MA Chidambaram Stadium"                        : (13.0635, 80.2790),
    "Chennai"                                       : (13.0635, 80.2790),

    # ── Delhi ─────────────────────────────────────────────────────────────────
    "Arun Jaitley Stadium"                          : (28.6366, 77.2306),
    "Feroz Shah Kotla"                              : (28.6366, 77.2306),
    "Delhi"                                         : (28.6366, 77.2306),

    # ── Hyderabad ─────────────────────────────────────────────────────────────
    "Rajiv Gandhi International Stadium"            : (17.4040, 78.5508),
    "Hyderabad"                                     : (17.4040, 78.5508),

    # ── Punjab / Mohali / New Chandigarh ──────────────────────────────────────
    "Punjab Cricket Association IS Bindra Stadium"  : (30.7046, 76.7179),
    "Maharaja Yadavindra Singh International Cricket Stadium" : (30.7700, 76.6460),
    "Mohali"                                        : (30.7046, 76.7179),

    # ── Jaipur ────────────────────────────────────────────────────────────────
    "Sawai Mansingh Stadium"                        : (26.8887, 75.8076),
    "Jaipur"                                        : (26.8887, 75.8076),

    # ── Ahmedabad / Motera ────────────────────────────────────────────────────
    "Narendra Modi Stadium"                         : (23.0916, 72.5938),
    "Ahmedabad"                                     : (23.0916, 72.5938),

    # ── Pune ──────────────────────────────────────────────────────────────────
    "Maharashtra Cricket Association Stadium"       : (18.6279, 73.8007),
    "Pune"                                          : (18.6279, 73.8007),

    # ── Indore ────────────────────────────────────────────────────────────────
    "Holkar Cricket Stadium"                        : (22.7196, 75.8577),
    "Indore"                                        : (22.7196, 75.8577),

    # ── Ranchi ────────────────────────────────────────────────────────────────
    "JSCA International Stadium Complex"            : (23.3441, 85.3096),
    "Ranchi"                                        : (23.3441, 85.3096),

    # ── Dharamsala ────────────────────────────────────────────────────────────
    "Himachal Pradesh Cricket Association Stadium"  : (32.2190, 76.3234),
    "Dharamsala"                                    : (32.2190, 76.3234),

    # ── Nagpur ────────────────────────────────────────────────────────────────
    "Vidarbha Cricket Association Stadium"          : (21.1080, 79.0750),
    "Nagpur"                                        : (21.1080, 79.0750),

    # ── Cuttack ───────────────────────────────────────────────────────────────
    "Barabati Stadium"                              : (20.4625, 85.8830),
    "Cuttack"                                       : (20.4625, 85.8830),

    # ── Visakhapatnam ─────────────────────────────────────────────────────────
    "Dr YS Rajasekhara Reddy ACA VDCA Cricket Stadium" : (17.7218, 83.3027),
    "Visakhapatnam"                                 : (17.7218, 83.3027),

    # ── Raipur ────────────────────────────────────────────────────────────────
    "Shaheed Veer Narayan Singh International Stadium" : (21.2849, 81.6298),
    "Raipur"                                        : (21.2849, 81.6298),

    # ── Rajkot ────────────────────────────────────────────────────────────────
    "Saurashtra Cricket Association Stadium"        : (22.2967, 70.7984),
    "Rajkot"                                        : (22.2967, 70.7984),

    # ── Kanpur ────────────────────────────────────────────────────────────────
    "Green Park"                                    : (26.4637, 80.3498),
    "Kanpur"                                        : (26.4637, 80.3498),

    # ── Lucknow ───────────────────────────────────────────────────────────────
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium" : (26.8928, 80.9461),
    "Ekana Cricket Stadium"                         : (26.8928, 80.9461),
    "Lucknow"                                       : (26.8928, 80.9461),

    # ── Guwahati ──────────────────────────────────────────────────────────────
    "Barsapara Cricket Stadium"                     : (26.1306, 91.7466),
    "Guwahati"                                      : (26.1306, 91.7466),

    # ── Nehru Stadium (Coimbatore) ────────────────────────────────────────────
    "Nehru Stadium"                                 : (11.0018, 76.9628),

    # ── South Africa ──────────────────────────────────────────────────────────
    "Newlands"                                      : (-33.9242, 18.4493),
    "Cape Town"                                     : (-33.9242, 18.4493),
    "St Georges Park"                               : (-33.9600, 25.5650),
    "Port Elizabeth"                                : (-33.9600, 25.5650),
    "Kingsmead"                                     : (-29.8574, 31.0215),
    "Durban"                                        : (-29.8574, 31.0215),
    "SuperSport Park"                               : (-25.7566, 28.2704),
    "Centurion"                                     : (-25.7566, 28.2704),
    "Buffalo Park"                                  : (-32.9900, 27.9050),
    "East London"                                   : (-32.9900, 27.9050),
    "New Wanderers Stadium"                         : (-26.1833, 28.0621),
    "Johannesburg"                                  : (-26.1833, 28.0621),
    "De Beers Diamond Oval"                         : (-28.7410, 24.7700),
    "Kimberley"                                     : (-28.7410, 24.7700),
    "OUTsurance Oval"                               : (-29.1041, 26.2143),
    "Bloemfontein"                                  : (-29.1041, 26.2143),

    # ── UAE ───────────────────────────────────────────────────────────────────
    "Dubai International Cricket Stadium"           : (25.0444, 55.3734),
    "Dubai"                                         : (25.0444, 55.3734),
    "Zayed Cricket Stadium"                         : (24.4212, 54.4771),
    "Sheikh Zayed Stadium"                          : (24.4212, 54.4771),
    "Abu Dhabi"                                     : (24.4212, 54.4771),
    "Sharjah Cricket Stadium"                       : (25.3373, 55.3893),
    "Sharjah"                                       : (25.3373, 55.3893),
}


# ── Stadium aliases ───────────────────────────────────────────────────────────
# Keys are normalised (lowercase, dots→spaces).
# Values are canonical VENUE_COORDS keys (exact capitalisation).
STADIUM_ALIASES = {
    # Renamed stadiums
    "sardar patel stadium"                           : "Narendra Modi Stadium",
    "sardar patel stadium motera"                    : "Narendra Modi Stadium",
    "sardar patel stadium, motera"                   : "Narendra Modi Stadium",
    "gujarat stadium"                                : "Narendra Modi Stadium",
    "subrata roy sahara stadium"                     : "Maharashtra Cricket Association Stadium",
    "sahara stadium"                                 : "Maharashtra Cricket Association Stadium",
    "feroz shah kotla"                               : "Arun Jaitley Stadium",
    "ferozeshah kotla"                               : "Arun Jaitley Stadium",

    # Punctuation / spacing variants
    "m chinnaswamy stadium"                          : "M Chinnaswamy Stadium",
    "mchinnaswamy stadium"                           : "M Chinnaswamy Stadium",

    # Long official names → short canonical
    "bharat ratna shri atal bihari vajpayee ekana cricket stadium"
                                                     : "Ekana Cricket Stadium",
    "ekana cricket stadium lucknow"                  : "Ekana Cricket Stadium",
    "dr ys rajasekhara reddy aca vdca cricket stadium"
                                                     : "Dr YS Rajasekhara Reddy ACA VDCA Cricket Stadium",
    "dr y s rajasekhara reddy aca vdca cricket stadium"
                                                     : "Dr YS Rajasekhara Reddy ACA VDCA Cricket Stadium",
    "aca vdca cricket stadium"                       : "Dr YS Rajasekhara Reddy ACA VDCA Cricket Stadium",
    "dr ys rajasekhara reddy aca vdca cricket stadium visakhapatnam"
                                                     : "Dr YS Rajasekhara Reddy ACA VDCA Cricket Stadium",
    "vidarbha cricket association stadium jamtha"    : "Vidarbha Cricket Association Stadium",
    "vidarbha cricket association stadium nagpur"    : "Vidarbha Cricket Association Stadium",
    "maharaja yadavindra singh international cricket stadium mullanpur"
                                                     : "Maharaja Yadavindra Singh International Cricket Stadium",
    "maharaja yadavindra singh international cricket stadium new chandigarh"
                                                     : "Maharaja Yadavindra Singh International Cricket Stadium",
    "barsapara cricket stadium guwahati"             : "Barsapara Cricket Stadium",
    "dr dy patil sports academy mumbai"              : "Dr DY Patil Sports Academy",
    "dr dy patil sports academy navi mumbai"         : "Dr DY Patil Sports Academy",
    "brabourne stadium mumbai"                       : "Brabourne Stadium",
    "ma chidambaram stadium chepauk"                 : "MA Chidambaram Stadium",
    "ma chidambaram stadium chepauk chennai"         : "MA Chidambaram Stadium",
    "rajiv gandhi international stadium uppal"       : "Rajiv Gandhi International Stadium",
    "rajiv gandhi international stadium uppal hyderabad"
                                                     : "Rajiv Gandhi International Stadium",
    "punjab cricket association is bindra stadium mohali"
                                                     : "Punjab Cricket Association IS Bindra Stadium",
    "punjab cricket association stadium mohali"      : "Punjab Cricket Association IS Bindra Stadium",
    "himachal pradesh cricket association stadium dharamsala"
                                                     : "Himachal Pradesh Cricket Association Stadium",
    "jsca international stadium complex ranchi"      : "JSCA International Stadium Complex",
    "wankhede stadium mumbai"                        : "Wankhede Stadium",
    "narendra modi stadium ahmedabad"                : "Narendra Modi Stadium",
    "maharashtra cricket association stadium pune"   : "Maharashtra Cricket Association Stadium",
    "holkar cricket stadium indore"                  : "Holkar Cricket Stadium",
    "sawai mansingh stadium jaipur"                  : "Sawai Mansingh Stadium",
    "eden gardens kolkata"                           : "Eden Gardens",

    # South Africa variants
    "st georges park"                                : "St Georges Park",
    "st george s park"                               : "St Georges Park",
    "outsurance oval"                                : "OUTsurance Oval",

    # UAE variants
    "zayed cricket stadium abu dhabi"                : "Zayed Cricket Stadium",
    "sheikh zayed stadium"                           : "Zayed Cricket Stadium",
    "dubai international cricket stadium dubai"      : "Dubai International Cricket Stadium",
    "sharjah cricket stadium sharjah"                : "Sharjah Cricket Stadium",
}


def _normalise(s: str) -> str:
    """Lowercase, remove apostrophes, replace dots/hyphens with spaces, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"['\u2019]", "", s)      # apostrophes removed (St George's → St Georges)
    s = re.sub(r"[.\-\u2013]", " ", s)   # dots, hyphens, en-dashes → spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def get_coords(venue: str):
    """
    Return (lat, lon) for a venue string. Resolution order:
    1. Exact normalised match against VENUE_COORDS keys
    2. Alias lookup
    3. Substring match (normalised, both directions)
    Returns None only if all four steps fail.
    """
    v_norm = _normalise(venue)

    # 1. Exact normalised match
    for key, coords in VENUE_COORDS.items():
        if _normalise(key) == v_norm:
            return coords

    # 2. Alias lookup
    if v_norm in STADIUM_ALIASES:
        canonical = STADIUM_ALIASES[v_norm]
        if canonical in VENUE_COORDS:
            return VENUE_COORDS[canonical]

    # 3 & 4. Substring match (normalised)
    for key, coords in VENUE_COORDS.items():
        k_norm = _normalise(key)
        if k_norm in v_norm or v_norm in k_norm:
            return coords

    return None


def fetch_weather_for_match(date: str, lat: float, lon: float,
                             retries: int = 3) -> dict:
    """Call Open-Meteo archive API. Returns dict of weather variables."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude"  : lat,
        "longitude" : lon,
        "start_date": date,
        "end_date"  : date,
        "daily"     : "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "hourly"    : "relativehumidity_2m,cloudcover,dewpoint_2m",
        "timezone"  : "Asia/Kolkata",
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                data   = r.json()
                daily  = data.get("daily",  {})
                hourly = data.get("hourly", {})

                temp_max = (daily.get("temperature_2m_max") or [None])[0]
                temp_min = (daily.get("temperature_2m_min") or [None])[0]
                precip   = (daily.get("precipitation_sum")  or [None])[0]
                wind_max = (daily.get("windspeed_10m_max")  or [None])[0]

                hum = hourly.get("relativehumidity_2m", [])
                cc  = hourly.get("cloudcover", [])
                dew = hourly.get("dewpoint_2m", [])

                eve_hum = np.mean(hum[17:24]) if len(hum) >= 24 else (np.mean(hum) if hum else None)
                eve_cc  = np.mean(cc[17:24])  if len(cc)  >= 24 else (np.mean(cc)  if cc  else None)
                eve_dew = np.mean(dew[17:24]) if len(dew) >= 24 else (np.mean(dew) if dew else None)

                return {
                    "temp_max_c"         : temp_max,
                    "temp_min_c"         : temp_min,
                    "precipitation_mm"   : precip,
                    "wind_speed_max_kmh" : wind_max,
                    "humidity_eve_pct"   : eve_hum,
                    "cloud_cover_eve_pct": eve_cc,
                    "dew_point_eve_c"    : eve_dew,
                }
            else:
                print(f"  HTTP {r.status_code} on attempt {attempt+1}")
        except Exception as e:
            print(f"  Error attempt {attempt+1}: {e}")
        time.sleep(2 ** attempt)
    return {}


def build_weather_table(matches_csv: str = "data/ipl_matches.csv",
                        output_csv:  str = "data/weather.csv",
                        pause_seconds: float = 0.3) -> pd.DataFrame:
    """Fetch weather for all matches. Resume-safe."""
    matches = pd.read_csv(matches_csv, parse_dates=["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    try:
        existing = pd.read_csv(output_csv)
        done_ids = set(existing["match_id"].astype(str))
        print(f"Resuming — {len(done_ids)} matches already fetched.")
    except FileNotFoundError:
        existing = pd.DataFrame()
        done_ids = set()

    results       = []
    total         = len(matches)
    unknown_venues = {}

    for i, row in matches.iterrows():
        mid    = str(row["match_id"])
        if mid in done_ids:
            continue

        venue  = str(row.get("venue", ""))
        date   = row["date"].strftime("%Y-%m-%d")
        coords = get_coords(venue)

        if coords is None:
            unknown_venues[venue] = unknown_venues.get(venue, 0) + 1
            results.append({"match_id": mid, "date": date, "venue": venue,
                             "weather_found": 0})
            continue

        lat, lon = coords
        weather  = fetch_weather_for_match(date, lat, lon)
        rec = {"match_id": mid, "date": date, "venue": venue,
               "lat": lat, "lon": lon, "weather_found": 1}
        rec.update(weather)
        results.append(rec)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] {venue[:45]} → {rec.get('temp_max_c','?')}°C")

        time.sleep(pause_seconds)

    new_df = pd.DataFrame(results)
    final  = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    final.to_csv(output_csv, index=False)

    found = int(final["weather_found"].sum()) if "weather_found" in final.columns else 0
    print(f"\nSaved {len(final)} rows to {output_csv}")
    print(f"Weather found for {found}/{len(final)} matches ({found/len(final)*100:.1f}%)")

    if unknown_venues:
        print(f"\nStill-unknown venues ({len(unknown_venues)}) — add to VENUE_COORDS:")
        for v, count in sorted(unknown_venues.items(), key=lambda x: -x[1]):
            print(f"  [{count:3d} matches]  '{v}'")

    return final


if __name__ == "__main__":
    import sys
    from pathlib import Path
    Path("data").mkdir(exist_ok=True)

    # ── Self-test matching before any network calls ───────────────────────────
    print("── Matching self-test ──────────────────────────────────────────────")
    tests = [
        ("M.Chinnaswamy Stadium",
                "M Chinnaswamy Stadium"),
        ("Sardar Patel Stadium, Motera",
                "Narendra Modi Stadium"),
        ("Zayed Cricket Stadium, Abu Dhabi",
                "Zayed Cricket Stadium"),
        ("Subrata Roy Sahara Stadium",
                "Maharashtra Cricket Association Stadium"),
        ("Feroz Shah Kotla",
                "Arun Jaitley Stadium"),
        ("Brabourne Stadium",
                "Brabourne Stadium"),
        ("Vidarbha Cricket Association Stadium, Jamtha",
                "Vidarbha Cricket Association Stadium"),
        ("Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
                "Ekana Cricket Stadium"),
        ("Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur",
                "Maharaja Yadavindra Singh International Cricket Stadium"),
        ("Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh",
                "Maharaja Yadavindra Singh International Cricket Stadium"),
        ("Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
                "Dr YS Rajasekhara Reddy ACA VDCA Cricket Stadium"),
        ("Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
                "Dr YS Rajasekhara Reddy ACA VDCA Cricket Stadium"),
        ("Newlands",                    "Newlands"),
        ("St George's Park",            "St Georges Park"),
        ("Kingsmead",                   "Kingsmead"),
        ("SuperSport Park",             "SuperSport Park"),
        ("OUTsurance Oval",             "OUTsurance Oval"),
        ("Buffalo Park",                "Buffalo Park"),
        ("New Wanderers Stadium",       "New Wanderers Stadium"),
        ("De Beers Diamond Oval",       "De Beers Diamond Oval"),
        ("MA Chidambaram Stadium, Chepauk, Chennai", "MA Chidambaram Stadium"),
        ("Dr DY Patil Sports Academy",  "Dr DY Patil Sports Academy"),
        ("Barsapara Cricket Stadium, Guwahati", "Barsapara Cricket Stadium"),
        ("Green Park",                  "Green Park"),
        ("Barabati Stadium",            "Barabati Stadium"),
        ("Saurashtra Cricket Association Stadium", "Saurashtra Cricket Association Stadium"),
        ("Shaheed Veer Narayan Singh International Stadium",
                "Shaheed Veer Narayan Singh International Stadium"),
    ]
    all_ok = True
    for venue, expected_key in tests:
        got    = get_coords(venue)
        expect = VENUE_COORDS.get(expected_key)
        ok     = (got == expect)
        if not ok:
            all_ok = False
            print(f"  ✗  '{venue}'")
            print(f"       expected {expected_key} → {expect}")
            print(f"       got                    → {got}")
        else:
            print(f"  ✓  '{venue[:65]}'")

    print(f"\nSelf-test {'PASSED ✓' if all_ok else 'FAILED ✗ — fix above before running'}")
    if not all_ok:
        sys.exit(1)

    print(f"\nFetching weather for all matches (~5 min)...")
    df = build_weather_table()
    print("\nDone.")
