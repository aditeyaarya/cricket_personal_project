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

    # ── Nepal ─────────────────────────────────────────────────────────────────
    "Tribhuvan University International Cricket Ground": (27.6915, 85.3206),
    "Kirtipur"                                         : (27.6915, 85.3206),

    # ── Oman ──────────────────────────────────────────────────────────────────
    "Al Amerat Cricket Ground Oman Cricket (Ministry Turf 1)": (23.6102, 58.5922),
    "Al Amerat Cricket Ground Oman Cricket (Ministry Turf 2)": (23.6102, 58.5922),
    "Oman Cricket Academy Ground"                      : (23.6102, 58.5922),

    # ── Australia ─────────────────────────────────────────────────────────────
    "Adelaide Oval"                                    : (-34.9155, 138.5960),
    "Adelaide"                                         : (-34.9155, 138.5960),
    "Melbourne Cricket Ground"                         : (-37.8200, 144.9836),
    "Melbourne"                                        : (-37.8200, 144.9836),
    "Sydney Cricket Ground"                            : (-33.8914, 151.2248),
    "Sydney"                                           : (-33.8914, 151.2248),
    "Perth Stadium"                                    : (-31.9505, 115.8890),
    "Perth"                                            : (-31.9505, 115.8890),
    "Western Australia Cricket Association Ground"     : (-31.9597, 115.8700),
    "WACA Ground"                                      : (-31.9597, 115.8700),
    "Brisbane Cricket Ground, Woolloongabba"           : (-27.4858, 153.0381),
    "The Gabba"                                        : (-27.4858, 153.0381),
    "Bellerive Oval"                                   : (-42.8794, 147.3574),
    "Hobart"                                           : (-42.8794, 147.3574),
    "Manuka Oval"                                      : (-35.3200, 149.1411),
    "Canberra"                                         : (-35.3200, 149.1411),
    "Carrara Oval"                                     : (-28.0027, 153.3808),
    "Stadium Australia"                                : (-33.8647, 151.2031),
    "GMHBA Stadium, South Geelong"                     : (-38.1527, 144.3577),
    "Simonds Stadium, South Geelong"                   : (-38.1527, 144.3577),
    "Marrara Stadium, Darwin"                          : (-12.3870, 130.8694),
    "Cazaly's Stadium, Cairns"                         : (-16.9186, 145.7781),

    # ── New Zealand ───────────────────────────────────────────────────────────
    "Eden Park"                                        : (-36.8760, 174.7436),
    "Auckland"                                         : (-36.8760, 174.7436),
    "Hagley Oval"                                      : (-43.5309, 172.6218),
    "Christchurch"                                     : (-43.5309, 172.6218),
    "Sky Stadium"                                      : (-41.2765, 174.8042),
    "Wellington"                                       : (-41.2765, 174.8042),
    "Westpac Stadium"                                  : (-41.2765, 174.8042),
    "McLean Park"                                      : (-39.5000, 176.8833),
    "Napier"                                           : (-39.5000, 176.8833),
    "Seddon Park"                                      : (-37.7878, 175.2793),
    "Hamilton"                                         : (-37.7878, 175.2793),
    "Bay Oval"                                         : (-37.6561, 176.1712),
    "Mount Maunganui"                                  : (-37.6561, 176.1712),
    "Saxton Oval"                                      : (-41.2863, 173.2449),
    "Nelson"                                           : (-41.2863, 173.2449),
    "University Oval"                                  : (-45.8788, 170.5028),
    "Dunedin"                                          : (-45.8788, 170.5028),
    "John Davies Oval, Queenstown"                     : (-45.0312, 168.6626),

    # ── England ───────────────────────────────────────────────────────────────
    "Edgbaston"                                        : (52.4556, -1.9022),
    "Birmingham"                                       : (52.4556, -1.9022),
    "Lord's"                                           : (51.5293, -0.1727),
    "London"                                           : (51.5293, -0.1727),
    "Kennington Oval"                                  : (51.4816, -0.1149),
    "The Oval"                                         : (51.4816, -0.1149),
    "Old Trafford"                                     : (53.4568, -2.2915),
    "Manchester"                                       : (53.4568, -2.2915),
    "Headingley"                                       : (53.8170, -1.5822),
    "Leeds"                                            : (53.8170, -1.5822),
    "Trent Bridge"                                     : (52.9363, -1.1325),
    "Nottingham"                                       : (52.9363, -1.1325),
    "The Rose Bowl"                                    : (50.9244, -1.3222),
    "Southampton"                                      : (50.9244, -1.3222),
    "Sophia Gardens"                                   : (51.4911, -3.1897),
    "Cardiff"                                          : (51.4911, -3.1897),
    "Riverside Ground"                                 : (54.8596, -1.5777),
    "Chester-le-Street"                                : (54.8596, -1.5777),
    "County Ground, Bristol"                           : (51.4555, -2.6026),
    "Bristol"                                          : (51.4555, -2.6026),

    # ── Ireland ───────────────────────────────────────────────────────────────
    "Malahide"                                         : (53.4503, -6.1541),
    "The Village, Malahide"                            : (53.4503, -6.1541),
    "Castle Avenue, Dublin"                            : (53.4503, -6.1541),
    "Civil Service Cricket Club, Stormont"             : (54.5921, -5.8765),
    "Belfast"                                          : (54.5921, -5.8765),

    # ── Scotland ──────────────────────────────────────────────────────────────
    "Grange Cricket Club Ground, Raeburn Place"        : (55.9534, -3.2196),
    "Edinburgh"                                        : (55.9534, -3.2196),
    "Titwood"                                          : (55.8352, -4.2896),
    "Glasgow"                                          : (55.8352, -4.2896),

    # ── Northern Ireland / Bready ─────────────────────────────────────────────
    "Bready Cricket Club, Magheramason"                : (54.9500, -7.4833),
    "Bready"                                           : (54.9500, -7.4833),

    # ── Netherlands ───────────────────────────────────────────────────────────
    "Sportpark Westvliet"                              : (52.0480, 4.2945),
    "The Hague"                                        : (52.0480, 4.2945),
    "Hazelaarweg"                                      : (51.8979, 4.4923),
    "VRA Ground"                                       : (52.3566, 4.8469),
    "Sportpark Het Schootsveld"                        : (52.2292, 6.8964),

    # ── West Indies / Caribbean ───────────────────────────────────────────────
    "Kensington Oval, Bridgetown"                      : (13.0969, -59.6145),
    "Bridgetown"                                       : (13.0969, -59.6145),
    "Barbados"                                         : (13.0969, -59.6145),
    "Sir Vivian Richards Stadium, North Sound"         : (17.1274, -61.8468),
    "Antigua"                                          : (17.1274, -61.8468),
    "Coolidge Cricket Ground, Antigua"                 : (17.1170, -61.7961),
    "Beausejour Stadium, Gros Islet"                   : (14.0810, -60.9493),
    "Daren Sammy National Cricket Stadium, Gros Islet" : (14.0810, -60.9493),
    "Darren Sammy National Cricket Stadium"            : (14.0810, -60.9493),
    "St Lucia"                                         : (14.0810, -60.9493),
    "Queen's Park Oval, Port of Spain"                 : (10.6430, -61.5267),
    "Port of Spain"                                    : (10.6430, -61.5267),
    "Trinidad"                                         : (10.6430, -61.5267),
    "Brian Lara Stadium, Tarouba"                      : (10.3250, -61.4524),
    "Providence Stadium"                               : (6.8013, -58.1551),
    "Guyana"                                           : (6.8013, -58.1551),
    "Sabina Park, Kingston"                            : (17.9963, -76.7956),
    "Kingston"                                         : (17.9963, -76.7956),
    "Jamaica"                                          : (17.9963, -76.7956),
    "Warner Park, Basseterre"                          : (17.3026, -62.7177),
    "St Kitts"                                         : (17.3026, -62.7177),
    "Arnos Vale Ground, Kingstown"                     : (13.1600, -61.2300),
    "St Vincent"                                       : (13.1600, -61.2300),
    "National Cricket Stadium, Grenada"                : (12.1165, -61.6790),
    "Grenada"                                          : (12.1165, -61.6790),
    "Windsor Park, Roseau"                             : (15.3017, -61.3883),
    "Dominica"                                         : (15.3017, -61.3883),
    "Central Broward Regional Park Stadium Turf Ground": (26.2009, -80.2550),
    "Lauderhill"                                       : (26.2009, -80.2550),
    "Nassau County International Cricket Stadium"      : (40.7282, -73.5950),
    "New York"                                         : (40.7282, -73.5950),
    "Grand Prairie Stadium, Dallas"                    : (32.6960, -97.0200),
    "Prairie View Cricket Complex"                     : (30.0910, -95.9870),

    # ── Pakistan ──────────────────────────────────────────────────────────────
    "Gaddafi Stadium"                                  : (31.5204, 74.3587),
    "Lahore"                                           : (31.5204, 74.3587),
    "National Stadium, Karachi"                        : (24.8918, 67.0640),
    "Karachi"                                          : (24.8918, 67.0640),
    "Rawalpindi Cricket Stadium"                       : (33.5651, 73.0169),
    "Rawalpindi"                                       : (33.5651, 73.0169),
    "Gymkhana Club Ground"                             : (33.7294, 73.0931),
    "Islamabad"                                        : (33.7294, 73.0931),

    # ── Sri Lanka ─────────────────────────────────────────────────────────────
    "R Premadasa Stadium"                              : (6.9366, 79.8608),
    "R.Premadasa Stadium, Khettarama"                  : (6.9366, 79.8608),
    "Colombo (RPS)"                                    : (6.9366, 79.8608),
    "Sinhalese Sports Club Ground, Colombo"            : (6.9020, 79.8607),
    "Colombo (SSC)"                                    : (6.9020, 79.8607),
    "Pallekele International Cricket Stadium"          : (7.2906, 80.5947),
    "Kandy"                                            : (7.2906, 80.5947),
    "Mahinda Rajapaksa International Cricket Stadium"  : (6.0367, 80.9659),
    "Sooriyawewa"                                      : (6.0367, 80.9659),
    "Rangiri Dambulla International Stadium"           : (7.8742, 80.6511),
    "Dambulla"                                         : (7.8742, 80.6511),

    # ── Bangladesh ────────────────────────────────────────────────────────────
    "Shere Bangla National Stadium"                    : (23.8103, 90.3600),
    "Mirpur"                                           : (23.8103, 90.3600),
    "Zahur Ahmed Chowdhury Stadium"                    : (22.3475, 91.8123),
    "Chattogram"                                       : (22.3475, 91.8123),
    "Bir Sreshtho Flight Lieutenant Matiur Rahman Stadium": (22.3475, 91.8123),
    "Sylhet International Cricket Stadium"             : (24.9045, 91.8611),
    "Sylhet Stadium"                                   : (24.9045, 91.8611),
    "Sheikh Abu Naser Stadium"                         : (22.8230, 89.5403),
    "Khulna"                                           : (22.8230, 89.5403),
    "Indian Association Ground"                        : (23.7104, 90.4074),

    # ── Zimbabwe ──────────────────────────────────────────────────────────────
    "Harare Sports Club"                               : (-17.8157, 31.0467),
    "Harare"                                           : (-17.8157, 31.0467),
    "Queens Sports Club, Bulawayo"                     : (-20.1667, 28.5667),
    "Bulawayo"                                         : (-20.1667, 28.5667),

    # ── Namibia ───────────────────────────────────────────────────────────────
    "Namibia Cricket Ground, Windhoek"                 : (-22.5597, 17.0832),
    "United Cricket Club Ground, Windhoek"             : (-22.5597, 17.0832),
    "Wanderers Cricket Ground, Windhoek"               : (-22.5597, 17.0832),
    "Windhoek"                                         : (-22.5597, 17.0832),

    # ── India (additional) ────────────────────────────────────────────────────
    "Greenfield International Stadium"                 : (8.5241, 76.9366),
    "Thiruvananthapuram"                               : (8.5241, 76.9366),
    "Shrimant Madhavrao Scindia Cricket Stadium"       : (26.2183, 78.1828),
    "Gwalior"                                          : (26.2183, 78.1828),

    # ── South Africa (additional) ─────────────────────────────────────────────
    "Boland Park"                                      : (-33.7249, 18.9543),
    "Paarl"                                            : (-33.7249, 18.9543),
    "Mangaung Oval"                                    : (-29.1214, 26.2154),
    "Bloemfontein (Mangaung)"                          : (-29.1214, 26.2154),
    "Senwes Park"                                      : (-26.6861, 27.0947),
    "Potchefstroom"                                    : (-26.6861, 27.0947),
    "Moses Mabhida Stadium"                            : (-29.8281, 31.0308),
    "The Wanderers Stadium"                            : (-26.1833, 28.0621),

    # ── Canada ────────────────────────────────────────────────────────────────
    "Maple Leaf North-West Ground, King City"          : (43.9285, -79.5386),
    "King City"                                        : (43.9285, -79.5386),

    # ── Papua New Guinea ──────────────────────────────────────────────────────
    "Amini Park, Port Moresby"                         : (-9.4438, 147.1803),
    "Port Moresby"                                     : (-9.4438, 147.1803),

    # ── Ghana ─────────────────────────────────────────────────────────────────
    "Achimota Senior Secondary School"                 : (5.6037, -0.2260),
    "Accra"                                            : (5.6037, -0.2260),

    # ── AMI Stadium (Christchurch — used before Hagley Oval) ─────────────────
    "AMI Stadium"                                      : (-43.5254, 172.5768),

    # ── Jade Stadium (Christchurch — old name) ────────────────────────────────
    "Jade Stadium"                                     : (-43.5254, 172.5768),

    # ── ICC Academy (Dubai) ───────────────────────────────────────────────────
    "ICC Academy"                                      : (25.0860, 55.1562),
    "7he Sevens Stadium, Dubai"                        : (25.0860, 55.1562),
    "Tolerance Oval"                                   : (24.4539, 54.3773),

    # ── Zhejiang (China) ─────────────────────────────────────────────────────
    "Zhejiang University of Technology Cricket Field"  : (30.3100, 120.0900),
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

    # Australia variants
    "woolloongabba"                                  : "Brisbane Cricket Ground, Woolloongabba",
    "the gabba"                                      : "Brisbane Cricket Ground, Woolloongabba",
    "brisbane cricket ground"                        : "Brisbane Cricket Ground, Woolloongabba",
    "waca ground"                                    : "Western Australia Cricket Association Ground",
    "waca"                                           : "Western Australia Cricket Association Ground",
    "western australia cricket association ground perth" : "Western Australia Cricket Association Ground",
    "scg"                                            : "Sydney Cricket Ground",
    "mcg"                                            : "Melbourne Cricket Ground",
    "adelaide oval adelaide"                         : "Adelaide Oval",
    "bellerive oval hobart"                          : "Bellerive Oval",
    "manuka oval canberra"                           : "Manuka Oval",
    "perth stadium perth"                            : "Perth Stadium",
    "stadium australia sydney"                       : "Stadium Australia",
    "gmhba stadium geelong"                          : "GMHBA Stadium, South Geelong",
    "simonds stadium geelong"                        : "Simonds Stadium, South Geelong",
    "cazalys stadium cairns"                         : "Cazaly's Stadium, Cairns",

    # New Zealand variants
    "eden park auckland"                             : "Eden Park",
    "hagley oval christchurch"                       : "Hagley Oval",
    "sky stadium wellington"                         : "Sky Stadium",
    "westpac stadium wellington"                     : "Sky Stadium",
    "basin reserve"                                  : "Sky Stadium",
    "mclean park napier"                             : "McLean Park",
    "seddon park hamilton"                           : "Seddon Park",
    "bay oval mount maunganui"                       : "Bay Oval",
    "saxton oval nelson"                             : "Saxton Oval",
    "university oval dunedin"                        : "University Oval",
    "auckland"                                       : "Eden Park",

    # England variants
    "edgbaston birmingham"                           : "Edgbaston",
    "lords"                                          : "Lord's",
    "lords cricket ground"                           : "Lord's",
    "the oval london"                                : "Kennington Oval",
    "kia oval"                                       : "Kennington Oval",
    "old trafford manchester"                        : "Old Trafford",
    "headingley leeds"                               : "Headingley",
    "headingley carnegie"                            : "Headingley",
    "trent bridge nottingham"                        : "Trent Bridge",
    "the rose bowl southampton"                      : "The Rose Bowl",
    "ageas bowl"                                     : "The Rose Bowl",
    "sophia gardens cardiff"                         : "Sophia Gardens",
    "riverside ground chester le street"             : "Riverside Ground",
    "county ground bristol"                          : "County Ground, Bristol",

    # Ireland variants
    "the village malahide dublin"                    : "The Village, Malahide",
    "the village malahide"                           : "The Village, Malahide",
    "malahide dublin"                                : "Malahide",
    "castle avenue dublin"                           : "Malahide",
    "civil service cricket club stormont belfast"    : "Civil Service Cricket Club, Stormont",
    "stormont belfast"                               : "Civil Service Cricket Club, Stormont",

    # Scotland variants
    "grange cricket club ground raeburn place edinburgh" : "Grange Cricket Club Ground, Raeburn Place",
    "raeburn place edinburgh"                        : "Grange Cricket Club Ground, Raeburn Place",
    "titwood glasgow"                                : "Titwood",

    # Netherlands variants
    "sportpark westvliet the hague"                  : "Sportpark Westvliet",
    "sportpark westvliet voorburg"                   : "Sportpark Westvliet",
    "hazelaarweg rotterdam"                          : "Hazelaarweg",
    "vra ground amstelveen"                          : "VRA Ground",

    # West Indies / Caribbean variants
    "kensington oval bridgetown barbados"            : "Kensington Oval, Bridgetown",
    "kensington oval"                                : "Kensington Oval, Bridgetown",
    "sir vivian richards stadium north sound antigua": "Sir Vivian Richards Stadium, North Sound",
    "beausejour stadium gros islet"                  : "Beausejour Stadium, Gros Islet",
    "daren sammy national cricket stadium gros islet st lucia" : "Daren Sammy National Cricket Stadium, Gros Islet",
    "darren sammy national cricket stadium st lucia" : "Darren Sammy National Cricket Stadium",
    "queens park oval port of spain"                 : "Queen's Park Oval, Port of Spain",
    "queens park oval port of spain trinidad"        : "Queen's Park Oval, Port of Spain",
    "brian lara stadium tarouba trinidad"            : "Brian Lara Stadium, Tarouba",
    "providence stadium guyana"                      : "Providence Stadium",
    "sabina park kingston jamaica"                   : "Sabina Park, Kingston",
    "warner park basseterre st kitts"                : "Warner Park, Basseterre",
    "arnos vale ground kingstown st vincent"         : "Arnos Vale Ground, Kingstown",
    "arnos vale ground kingstown"                    : "Arnos Vale Ground, Kingstown",
    "national cricket stadium st georges grenada"    : "National Cricket Stadium, Grenada",
    "windsor park roseau dominica"                   : "Windsor Park, Roseau",
    "central broward regional park stadium turf ground lauderhill" : "Central Broward Regional Park Stadium Turf Ground",
    "nassau county international cricket stadium new york" : "Nassau County International Cricket Stadium",
    "grand prairie stadium"                          : "Grand Prairie Stadium, Dallas",

    # Pakistan variants
    "gaddafi stadium lahore"                         : "Gaddafi Stadium",
    "national stadium karachi"                       : "National Stadium, Karachi",
    "rawalpindi cricket stadium rawalpindi"          : "Rawalpindi Cricket Stadium",

    # Sri Lanka variants
    "r premadasa stadium colombo"                    : "R Premadasa Stadium",
    "r premadasa stadium khettarama"                 : "R Premadasa Stadium",
    "rpremadasa stadium khettarama"                  : "R Premadasa Stadium",
    "sinhalese sports club ground"                   : "Sinhalese Sports Club Ground, Colombo",
    "pallekele international cricket stadium kandy"  : "Pallekele International Cricket Stadium",
    "mahinda rajapaksa international cricket stadium sooriyawewa" : "Mahinda Rajapaksa International Cricket Stadium",
    "rangiri dambulla international stadium"         : "Rangiri Dambulla International Stadium",

    # Bangladesh variants
    "shere bangla national stadium mirpur"           : "Shere Bangla National Stadium",
    "shere bangla national stadium mirpur dhaka"     : "Shere Bangla National Stadium",
    "zahur ahmed chowdhury stadium chattogram"       : "Zahur Ahmed Chowdhury Stadium",
    "zac stadium"                                    : "Zahur Ahmed Chowdhury Stadium",
    "bir sreshtho flight lieutenant matiur rahman stadium chattogram" : "Bir Sreshtho Flight Lieutenant Matiur Rahman Stadium",
    "sylhet international cricket stadium sylhet"    : "Sylhet International Cricket Stadium",
    "sheikh abu naser stadium khulna"                : "Sheikh Abu Naser Stadium",

    # Zimbabwe variants
    "harare sports club harare"                      : "Harare Sports Club",
    "queens sports club bulawayo"                    : "Queens Sports Club, Bulawayo",

    # Namibia variants
    "namibia cricket ground"                         : "Namibia Cricket Ground, Windhoek",
    "united cricket club ground"                     : "United Cricket Club Ground, Windhoek",
    "wanderers cricket ground windhoek"              : "Wanderers Cricket Ground, Windhoek",

    # India additional
    "greenfield international stadium thiruvananthapuram" : "Greenfield International Stadium",
    "shrimant madhavrao scindia cricket stadium gwalior"  : "Shrimant Madhavrao Scindia Cricket Stadium",

    # South Africa additional
    "boland park paarl"                              : "Boland Park",
    "mangaung oval bloemfontein"                     : "Mangaung Oval",
    "senwes park potchefstroom"                      : "Senwes Park",

    # Oman variants
    "al amerat cricket ground ministry turf 1"       : "Al Amerat Cricket Ground Oman Cricket (Ministry Turf 1)",
    "al amerat cricket ground ministry turf 2"       : "Al Amerat Cricket Ground Oman Cricket (Ministry Turf 2)",
    "oman cricket academy ground"                    : "Oman Cricket Academy Ground",

    # Nepal variants
    "tribhuvan university international cricket ground kirtipur" : "Tribhuvan University International Cricket Ground",

    # Christchurch old names
    "ami stadium christchurch"                       : "AMI Stadium",
    "jade stadium christchurch"                      : "Jade Stadium",

    # ICC / Dubai misc
    "icc academy dubai"                              : "ICC Academy",
    "7he sevens stadium dubai"                       : "ICC Academy",
    "tolerance oval abu dhabi"                       : "Tolerance Oval",
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


def build_weather_table(matches_csv: str = "data/all_matches.csv",
                        output_csv:  str = "data/weather.csv",
                        pause_seconds: float = 0.3) -> pd.DataFrame:
    """
    Fetch weather for all matches in matches_csv. Resume-safe.
    Defaults to data/all_matches.csv (multi-league) but falls back
    to data/ipl_matches.csv if the combined file does not exist.
    """
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