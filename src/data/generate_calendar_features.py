"""Calendar and Seasonal Feature Generator for BusDemand AI.

This module generates a reproducible 3-year daily calendar reference dataset
(2024-01-01 to 2026-12-31, exactly 1,096 dates) and an official holiday events
registry for the three corridor states: Karnataka (KA), Kerala (KL), and Tamil Nadu (TN).

IMPORTANT:
- Official holidays are grounded in authentic state government gazette notifications.
- Vacation periods and seasonal classifications are documented scenario assumptions.
"""

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path when executed directly
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from src.config import RAW_DATA_DIR, SIMULATION_END_DATE, SIMULATION_START_DATE

# ---------------------------------------------------------------------------
# Official State Government & National Holiday Events Registry (2024-2026)
# Sourced from official State Government Gazettes and Department Orders.
# ---------------------------------------------------------------------------

OFFICIAL_HOLIDAY_EVENTS: List[Dict[str, Any]] = [
    # =========================================================================
    # YEAR 2024
    # =========================================================================
    # Tamil Nadu 2024 (G.O. (Ms.) No. 692, Public (Misc) Dept, 09.11.2023)
    {"holiday_id": "H_2024_TN_001", "date": "2024-01-01", "holiday_name": "New Year's Day", "state": "TN", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.692", "source_url": "https://www.tn.gov.in", "notes": "Public holiday under NI Act"},
    {"holiday_id": "H_2024_TN_002", "date": "2024-01-15", "holiday_name": "Pongal", "state": "TN", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.692", "source_url": "https://www.tn.gov.in", "notes": "Harvest festival peak mobility"},
    {"holiday_id": "H_2024_TN_003", "date": "2024-01-16", "holiday_name": "Thiruvalluvar Day", "state": "TN", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.692", "source_url": "https://www.tn.gov.in", "notes": "Pongal holiday sequence"},
    {"holiday_id": "H_2024_TN_004", "date": "2024-01-17", "holiday_name": "Uzhavar Thirunal", "state": "TN", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.692", "source_url": "https://www.tn.gov.in", "notes": "Farmers festival holiday"},
    {"holiday_id": "H_2024_TN_005", "date": "2024-01-25", "holiday_name": "Thai Poosam", "state": "TN", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.692", "source_url": "https://www.tn.gov.in", "notes": "Religious observance holiday"},
    {"holiday_id": "H_2024_TN_006", "date": "2024-03-29", "holiday_name": "Good Friday", "state": "TN", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.692", "source_url": "https://www.tn.gov.in", "notes": "Christian observance"},
    {"holiday_id": "H_2024_TN_007", "date": "2024-04-09", "holiday_name": "Telugu New Year's Day", "state": "TN", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.692", "source_url": "https://www.tn.gov.in", "notes": "Ugadi celebration"},
    {"holiday_id": "H_2024_TN_008", "date": "2024-04-11", "holiday_name": "Ramzan (Eid-ul-Fitr)", "state": "TN", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.692", "source_url": "https://www.tn.gov.in", "notes": "Subject to moon sighting"},
    {"holiday_id": "H_2024_TN_009", "date": "2024-05-01", "holiday_name": "May Day", "state": "TN", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.692", "source_url": "https://www.tn.gov.in", "notes": "International Labour Day"},
    {"holiday_id": "H_2024_TN_010", "date": "2024-06-17", "holiday_name": "Bakrid (Eid al-Adha)", "state": "TN", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.692", "source_url": "https://www.tn.gov.in", "notes": "Islamic calendar observance"},
    {"holiday_id": "H_2024_TN_011", "date": "2024-07-17", "holiday_name": "Muharram", "state": "TN", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.692", "source_url": "https://www.tn.gov.in", "notes": "Islamic New Year"},
    {"holiday_id": "H_2024_TN_012", "date": "2024-08-26", "holiday_name": "Krishna Jayanthi", "state": "TN", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.692", "source_url": "https://www.tn.gov.in", "notes": "Gokulashtami"},
    {"holiday_id": "H_2024_TN_013", "date": "2024-09-07", "holiday_name": "Vinayakar Chathurthi", "state": "TN", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.692", "source_url": "https://www.tn.gov.in", "notes": "Ganesh Chaturthi"},
    {"holiday_id": "H_2024_TN_014", "date": "2024-09-16", "holiday_name": "Milad-un-Nabi", "state": "TN", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.692", "source_url": "https://www.tn.gov.in", "notes": "Prophet Muhammad's Birthday"},
    {"holiday_id": "H_2024_TN_015", "date": "2024-10-11", "holiday_name": "Ayutha Pooja", "state": "TN", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.692", "source_url": "https://www.tn.gov.in", "notes": "Navaratri / Dussehra sequence"},
    {"holiday_id": "H_2024_TN_016", "date": "2024-10-31", "holiday_name": "Deepavali", "state": "TN", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.692", "source_url": "https://www.tn.gov.in", "notes": "Festival of Lights"},
    {"holiday_id": "H_2024_TN_017", "date": "2024-12-25", "holiday_name": "Christmas", "state": "TN", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.692", "source_url": "https://www.tn.gov.in", "notes": "Christmas Day"},

    # Karnataka 2024 (Notification No. DPAR 16 HHL 2023, 28.11.2023)
    {"holiday_id": "H_2024_KA_001", "date": "2024-01-15", "holiday_name": "Makara Sankranti", "state": "KA", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2023", "source_url": "https://dpar.karnataka.gov.in", "notes": "Harvest festival"},
    {"holiday_id": "H_2024_KA_002", "date": "2024-03-08", "holiday_name": "Maha Shivaratri", "state": "KA", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2023", "source_url": "https://dpar.karnataka.gov.in", "notes": "Religious holiday"},
    {"holiday_id": "H_2024_KA_003", "date": "2024-03-29", "holiday_name": "Good Friday", "state": "KA", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2023", "source_url": "https://dpar.karnataka.gov.in", "notes": "Christian observance"},
    {"holiday_id": "H_2024_KA_004", "date": "2024-04-09", "holiday_name": "Ugadi Festival", "state": "KA", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2023", "source_url": "https://dpar.karnataka.gov.in", "notes": "Kannada New Year"},
    {"holiday_id": "H_2024_KA_005", "date": "2024-04-11", "holiday_name": "Khutub-E-Ramzan", "state": "KA", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2023", "source_url": "https://dpar.karnataka.gov.in", "notes": "Id-ul-Fitr"},
    {"holiday_id": "H_2024_KA_006", "date": "2024-05-01", "holiday_name": "May Day", "state": "KA", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2023", "source_url": "https://dpar.karnataka.gov.in", "notes": "Labour Day"},
    {"holiday_id": "H_2024_KA_007", "date": "2024-05-10", "holiday_name": "Basava Jayanthi / Akshaya Tritiya", "state": "KA", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2023", "source_url": "https://dpar.karnataka.gov.in", "notes": "Philosopher Basavanna Birth Anniversary"},
    {"holiday_id": "H_2024_KA_008", "date": "2024-06-17", "holiday_name": "Bakrid", "state": "KA", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2023", "source_url": "https://dpar.karnataka.gov.in", "notes": "Id-ul-Adha"},
    {"holiday_id": "H_2024_KA_009", "date": "2024-07-17", "holiday_name": "Last Day of Moharram", "state": "KA", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2023", "source_url": "https://dpar.karnataka.gov.in", "notes": "Ashura observance"},
    {"holiday_id": "H_2024_KA_010", "date": "2024-09-07", "holiday_name": "Varasiddhi Vinayaka Vrata", "state": "KA", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2023", "source_url": "https://dpar.karnataka.gov.in", "notes": "Ganesh Chaturthi"},
    {"holiday_id": "H_2024_KA_011", "date": "2024-09-16", "holiday_name": "Eid-Milad", "state": "KA", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2023", "source_url": "https://dpar.karnataka.gov.in", "notes": "Milad-un-Nabi"},
    {"holiday_id": "H_2024_KA_012", "date": "2024-10-11", "holiday_name": "Mahanavami / Ayudha Pooja", "state": "KA", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2023", "source_url": "https://dpar.karnataka.gov.in", "notes": "Dasara festival peak"},
    {"holiday_id": "H_2024_KA_013", "date": "2024-10-17", "holiday_name": "Maharshi Valmiki Jayanthi", "state": "KA", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2023", "source_url": "https://dpar.karnataka.gov.in", "notes": "Sage Valmiki Birthday"},
    {"holiday_id": "H_2024_KA_014", "date": "2024-10-31", "holiday_name": "Naraka Chaturdashi", "state": "KA", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2023", "source_url": "https://dpar.karnataka.gov.in", "notes": "Diwali day 1"},
    {"holiday_id": "H_2024_KA_015", "date": "2024-11-01", "holiday_name": "Kannada Rajyotsava", "state": "KA", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2023", "source_url": "https://dpar.karnataka.gov.in", "notes": "Karnataka State Formation Day"},
    {"holiday_id": "H_2024_KA_016", "date": "2024-11-02", "holiday_name": "Balipadyami / Deepavali", "state": "KA", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2023", "source_url": "https://dpar.karnataka.gov.in", "notes": "Diwali day 2"},
    {"holiday_id": "H_2024_KA_017", "date": "2024-11-18", "holiday_name": "Kanakadasa Jayanthi", "state": "KA", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2023", "source_url": "https://dpar.karnataka.gov.in", "notes": "Saint Kanakadasa Birthday"},
    {"holiday_id": "H_2024_KA_018", "date": "2024-12-25", "holiday_name": "Christmas", "state": "KA", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2023", "source_url": "https://dpar.karnataka.gov.in", "notes": "Christmas Day"},

    # Kerala 2024 (G.O. (P) No. 24/2023/GAD, 04.10.2023)
    {"holiday_id": "H_2024_KL_001", "date": "2024-01-02", "holiday_name": "Mannam Jayanthi", "state": "KL", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.24/2023/GAD", "source_url": "https://kerala.gov.in", "notes": "Social reformer Mannathu Padmanabhan Birthday"},
    {"holiday_id": "H_2024_KL_002", "date": "2024-03-08", "holiday_name": "Sivarathri", "state": "KL", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.24/2023/GAD", "source_url": "https://kerala.gov.in", "notes": "Aluva Sivarathri festival"},
    {"holiday_id": "H_2024_KL_003", "date": "2024-03-28", "holiday_name": "Maundy Thursday", "state": "KL", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.24/2023/GAD", "source_url": "https://kerala.gov.in", "notes": "Holy Thursday"},
    {"holiday_id": "H_2024_KL_004", "date": "2024-03-29", "holiday_name": "Good Friday", "state": "KL", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.24/2023/GAD", "source_url": "https://kerala.gov.in", "notes": "Good Friday"},
    {"holiday_id": "H_2024_KL_005", "date": "2024-04-10", "holiday_name": "Id-ul-Fitr (Ramzan)", "state": "KL", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.24/2023/GAD", "source_url": "https://kerala.gov.in", "notes": "Observed on moon sighting"},
    {"holiday_id": "H_2024_KL_006", "date": "2024-05-01", "holiday_name": "May Day", "state": "KL", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.24/2023/GAD", "source_url": "https://kerala.gov.in", "notes": "Workers Day"},
    {"holiday_id": "H_2024_KL_007", "date": "2024-06-17", "holiday_name": "Bakrid (Id-ul-Adha)", "state": "KL", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.24/2023/GAD", "source_url": "https://kerala.gov.in", "notes": "Sacrifice feast"},
    {"holiday_id": "H_2024_KL_008", "date": "2024-07-16", "holiday_name": "Muharram", "state": "KL", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.24/2023/GAD", "source_url": "https://kerala.gov.in", "notes": "Islamic New Year day"},
    {"holiday_id": "H_2024_KL_009", "date": "2024-08-03", "holiday_name": "Karkkadaka Vaavu", "state": "KL", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.24/2023/GAD", "source_url": "https://kerala.gov.in", "notes": "Ancestral ritual day"},
    {"holiday_id": "H_2024_KL_010", "date": "2024-08-20", "holiday_name": "Sree Narayana Guru Jayanthi", "state": "KL", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.24/2023/GAD", "source_url": "https://kerala.gov.in", "notes": "SNDP Guru Jayanthi"},
    {"holiday_id": "H_2024_KL_011", "date": "2024-09-16", "holiday_name": "Milad-i-Sherif", "state": "KL", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.24/2023/GAD", "source_url": "https://kerala.gov.in", "notes": "Nabi Dinam"},
    {"holiday_id": "H_2024_KL_012", "date": "2024-10-31", "holiday_name": "Deepavali", "state": "KL", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.24/2023/GAD", "source_url": "https://kerala.gov.in", "notes": "Diwali"},
    {"holiday_id": "H_2024_KL_013", "date": "2024-12-25", "holiday_name": "Christmas", "state": "KL", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.24/2023/GAD", "source_url": "https://kerala.gov.in", "notes": "Christmas Day"},

    # National Holidays 2024
    {"holiday_id": "H_2024_NAT_001", "date": "2024-01-26", "holiday_name": "Republic Day", "state": "ALL", "year": 2024, "is_gazetted": True, "is_national": True, "verification_status": "verified_official_gazette", "source_name": "Ministry of Personnel, Govt of India", "source_url": "https://dopt.gov.in", "notes": "National holiday celebrated across all states"},
    {"holiday_id": "H_2024_NAT_002", "date": "2024-08-15", "holiday_name": "Independence Day", "state": "ALL", "year": 2024, "is_gazetted": True, "is_national": True, "verification_status": "verified_official_gazette", "source_name": "Ministry of Personnel, Govt of India", "source_url": "https://dopt.gov.in", "notes": "National holiday celebrated across all states"},
    {"holiday_id": "H_2024_NAT_003", "date": "2024-10-02", "holiday_name": "Mahatma Gandhi Jayanti", "state": "ALL", "year": 2024, "is_gazetted": True, "is_national": True, "verification_status": "verified_official_gazette", "source_name": "Ministry of Personnel, Govt of India", "source_url": "https://dopt.gov.in", "notes": "National holiday celebrated across all states"},

    # =========================================================================
    # YEAR 2025
    # =========================================================================
    # Tamil Nadu 2025 (G.O. (Ms.) No. 792, Public (Misc) Dept, 22.11.2024)
    {"holiday_id": "H_2025_TN_001", "date": "2025-01-14", "holiday_name": "Pongal", "state": "TN", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.792", "source_url": "https://www.tn.gov.in", "notes": "Thai Pongal harvest festival"},
    {"holiday_id": "H_2025_TN_002", "date": "2025-01-15", "holiday_name": "Thiruvalluvar Day", "state": "TN", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.792", "source_url": "https://www.tn.gov.in", "notes": "Pongal festival chain"},
    {"holiday_id": "H_2025_TN_003", "date": "2025-01-16", "holiday_name": "Uzhavar Thirunal", "state": "TN", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.792", "source_url": "https://www.tn.gov.in", "notes": "Farmers festival"},
    {"holiday_id": "H_2025_TN_004", "date": "2025-02-11", "holiday_name": "Thai Poosam", "state": "TN", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.792", "source_url": "https://www.tn.gov.in", "notes": "Murugan temple festivals"},
    {"holiday_id": "H_2025_TN_005", "date": "2025-03-31", "holiday_name": "Ramzan (Idul Fitr)", "state": "TN", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.792", "source_url": "https://www.tn.gov.in", "notes": "Subject to moon sighting"},
    {"holiday_id": "H_2025_TN_006", "date": "2025-04-10", "holiday_name": "Mahaveer Jayanthi", "state": "TN", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.792", "source_url": "https://www.tn.gov.in", "notes": "Jain observance"},
    {"holiday_id": "H_2025_TN_007", "date": "2025-04-14", "holiday_name": "Tamil New Year / Dr Ambedkar Birthday", "state": "TN", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.792", "source_url": "https://www.tn.gov.in", "notes": "Puthandu celebration"},
    {"holiday_id": "H_2025_TN_008", "date": "2025-04-18", "holiday_name": "Good Friday", "state": "TN", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.792", "source_url": "https://www.tn.gov.in", "notes": "Christian observance"},
    {"holiday_id": "H_2025_TN_009", "date": "2025-05-01", "holiday_name": "May Day", "state": "TN", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.792", "source_url": "https://www.tn.gov.in", "notes": "Labour Day"},
    {"holiday_id": "H_2025_TN_010", "date": "2025-06-07", "holiday_name": "Bakrid (Idul Azha)", "state": "TN", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.792", "source_url": "https://www.tn.gov.in", "notes": "Islamic feast"},
    {"holiday_id": "H_2025_TN_011", "date": "2025-08-16", "holiday_name": "Krishna Jayanthi", "state": "TN", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.792", "source_url": "https://www.tn.gov.in", "notes": "Janmashtami"},
    {"holiday_id": "H_2025_TN_012", "date": "2025-08-28", "holiday_name": "Vinayakar Chathurthi", "state": "TN", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.792", "source_url": "https://www.tn.gov.in", "notes": "Ganesh Chaturthi"},
    {"holiday_id": "H_2025_TN_013", "date": "2025-09-05", "holiday_name": "Milad-un-Nabi", "state": "TN", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.792", "source_url": "https://www.tn.gov.in", "notes": "Prophet Muhammad Birthday"},
    {"holiday_id": "H_2025_TN_014", "date": "2025-10-01", "holiday_name": "Ayudha Pooja", "state": "TN", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.792", "source_url": "https://www.tn.gov.in", "notes": "Navaratri holiday"},
    {"holiday_id": "H_2025_TN_015", "date": "2025-10-02", "holiday_name": "Vijaya Dasami", "state": "TN", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.792", "source_url": "https://www.tn.gov.in", "notes": "Conjoined with Gandhi Jayanti in TN"},
    {"holiday_id": "H_2025_TN_016", "date": "2025-10-20", "holiday_name": "Deepavali", "state": "TN", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.792", "source_url": "https://www.tn.gov.in", "notes": "Festival of Lights"},
    {"holiday_id": "H_2025_TN_017", "date": "2025-12-25", "holiday_name": "Christmas Day", "state": "TN", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.792", "source_url": "https://www.tn.gov.in", "notes": "Christmas"},

    # Karnataka 2025 (Notification No. DPAR 16 HHL 2024, 21.11.2024)
    {"holiday_id": "H_2025_KA_001", "date": "2025-01-14", "holiday_name": "Makara Sankranti", "state": "KA", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2024", "source_url": "https://dpar.karnataka.gov.in", "notes": "Harvest festival"},
    {"holiday_id": "H_2025_KA_002", "date": "2025-02-26", "holiday_name": "Maha Shivaratri", "state": "KA", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2024", "source_url": "https://dpar.karnataka.gov.in", "notes": "Shivaratri"},
    {"holiday_id": "H_2025_KA_003", "date": "2025-03-31", "holiday_name": "Khutub-E-Ramzan", "state": "KA", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2024", "source_url": "https://dpar.karnataka.gov.in", "notes": "Id-ul-Fitr"},
    {"holiday_id": "H_2025_KA_004", "date": "2025-04-10", "holiday_name": "Mahaveera Jayanthi", "state": "KA", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2024", "source_url": "https://dpar.karnataka.gov.in", "notes": "Jain observance"},
    {"holiday_id": "H_2025_KA_005", "date": "2025-04-14", "holiday_name": "Dr. B.R. Ambedkar Jayanthi", "state": "KA", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2024", "source_url": "https://dpar.karnataka.gov.in", "notes": "Ambedkar Birthday"},
    {"holiday_id": "H_2025_KA_006", "date": "2025-04-18", "holiday_name": "Good Friday", "state": "KA", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2024", "source_url": "https://dpar.karnataka.gov.in", "notes": "Christian observance"},
    {"holiday_id": "H_2025_KA_007", "date": "2025-04-30", "holiday_name": "Basava Jayanthi / Akshaya Tritiya", "state": "KA", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2024", "source_url": "https://dpar.karnataka.gov.in", "notes": "Basavanna Birthday"},
    {"holiday_id": "H_2025_KA_008", "date": "2025-05-01", "holiday_name": "May Day", "state": "KA", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2024", "source_url": "https://dpar.karnataka.gov.in", "notes": "Labour Day"},
    {"holiday_id": "H_2025_KA_009", "date": "2025-06-07", "holiday_name": "Bakrid (Id-ul-Zuha)", "state": "KA", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2024", "source_url": "https://dpar.karnataka.gov.in", "notes": "Sacrifice feast"},
    {"holiday_id": "H_2025_KA_010", "date": "2025-08-27", "holiday_name": "Varasiddhi Vinayaka Vrata", "state": "KA", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2024", "source_url": "https://dpar.karnataka.gov.in", "notes": "Ganesh Chaturthi"},
    {"holiday_id": "H_2025_KA_011", "date": "2025-09-05", "holiday_name": "Eid-Milad", "state": "KA", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2024", "source_url": "https://dpar.karnataka.gov.in", "notes": "Prophet Birthday"},
    {"holiday_id": "H_2025_KA_012", "date": "2025-10-01", "holiday_name": "Mahanavami / Ayudha Pooja", "state": "KA", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2024", "source_url": "https://dpar.karnataka.gov.in", "notes": "Dasara festival"},
    {"holiday_id": "H_2025_KA_013", "date": "2025-10-07", "holiday_name": "Maharshi Valmiki Jayanthi", "state": "KA", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2024", "source_url": "https://dpar.karnataka.gov.in", "notes": "Valmiki Birthday"},
    {"holiday_id": "H_2025_KA_014", "date": "2025-10-20", "holiday_name": "Naraka Chaturdashi", "state": "KA", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2024", "source_url": "https://dpar.karnataka.gov.in", "notes": "Diwali day 1"},
    {"holiday_id": "H_2025_KA_015", "date": "2025-10-22", "holiday_name": "Balipadyami / Deepavali", "state": "KA", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2024", "source_url": "https://dpar.karnataka.gov.in", "notes": "Diwali day 2"},
    {"holiday_id": "H_2025_KA_016", "date": "2025-11-01", "holiday_name": "Kannada Rajyotsava", "state": "KA", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2024", "source_url": "https://dpar.karnataka.gov.in", "notes": "Karnataka Formation Day"},
    {"holiday_id": "H_2025_KA_017", "date": "2025-12-25", "holiday_name": "Christmas", "state": "KA", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2024", "source_url": "https://dpar.karnataka.gov.in", "notes": "Christmas Day"},

    # Kerala 2025 (G.O. (P) No. 34/2024/GAD, 14.10.2024)
    {"holiday_id": "H_2025_KL_001", "date": "2025-01-02", "holiday_name": "Mannam Jayanthi", "state": "KL", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.34/2024/GAD", "source_url": "https://kerala.gov.in", "notes": "Mannam Jayanthi"},
    {"holiday_id": "H_2025_KL_002", "date": "2025-02-26", "holiday_name": "Maha Sivarathri", "state": "KL", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.34/2024/GAD", "source_url": "https://kerala.gov.in", "notes": "Sivarathri"},
    {"holiday_id": "H_2025_KL_003", "date": "2025-03-31", "holiday_name": "Id-ul-Fitr (Ramzan)", "state": "KL", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.34/2024/GAD", "source_url": "https://kerala.gov.in", "notes": "Subject to moon sighting"},
    {"holiday_id": "H_2025_KL_004", "date": "2025-04-14", "holiday_name": "Vishu / Dr. B.R. Ambedkar Jayanthi", "state": "KL", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.34/2024/GAD", "source_url": "https://kerala.gov.in", "notes": "Malayalam New Year harvest day"},
    {"holiday_id": "H_2025_KL_005", "date": "2025-04-17", "holiday_name": "Maundy Thursday", "state": "KL", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.34/2024/GAD", "source_url": "https://kerala.gov.in", "notes": "Holy Thursday"},
    {"holiday_id": "H_2025_KL_006", "date": "2025-04-18", "holiday_name": "Good Friday", "state": "KL", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.34/2024/GAD", "source_url": "https://kerala.gov.in", "notes": "Good Friday"},
    {"holiday_id": "H_2025_KL_007", "date": "2025-05-01", "holiday_name": "May Day", "state": "KL", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.34/2024/GAD", "source_url": "https://kerala.gov.in", "notes": "May Day"},
    {"holiday_id": "H_2025_KL_008", "date": "2025-06-06", "holiday_name": "Id-ul-Ad'ha (Bakrid)", "state": "KL", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.34/2024/GAD", "source_url": "https://kerala.gov.in", "notes": "Bakrid"},
    {"holiday_id": "H_2025_KL_009", "date": "2025-07-24", "holiday_name": "Karkadaka Vavu", "state": "KL", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.34/2024/GAD", "source_url": "https://kerala.gov.in", "notes": "Vavu Bali"},
    {"holiday_id": "H_2025_KL_010", "date": "2025-08-28", "holiday_name": "Ayyankali Jayanthi", "state": "KL", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.34/2024/GAD", "source_url": "https://kerala.gov.in", "notes": "Social leader Ayyankali Birthday"},
    {"holiday_id": "H_2025_KL_011", "date": "2025-09-04", "holiday_name": "First Onam", "state": "KL", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.34/2024/GAD", "source_url": "https://kerala.gov.in", "notes": "Onam celebration day 1"},
    {"holiday_id": "H_2025_KL_012", "date": "2025-09-05", "holiday_name": "Thiruvonam / Milad-i-Sherif", "state": "KL", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.34/2024/GAD", "source_url": "https://kerala.gov.in", "notes": "Main Onam day"},
    {"holiday_id": "H_2025_KL_013", "date": "2025-09-06", "holiday_name": "Third Onam", "state": "KL", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.34/2024/GAD", "source_url": "https://kerala.gov.in", "notes": "Avittom day"},
    {"holiday_id": "H_2025_KL_014", "date": "2025-10-01", "holiday_name": "Mahanavami", "state": "KL", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.34/2024/GAD", "source_url": "https://kerala.gov.in", "notes": "Puja holiday"},
    {"holiday_id": "H_2025_KL_015", "date": "2025-10-02", "holiday_name": "Vijayadasami", "state": "KL", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.34/2024/GAD", "source_url": "https://kerala.gov.in", "notes": "Vidyarambham day"},
    {"holiday_id": "H_2025_KL_016", "date": "2025-10-20", "holiday_name": "Deepavali", "state": "KL", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.34/2024/GAD", "source_url": "https://kerala.gov.in", "notes": "Diwali"},
    {"holiday_id": "H_2025_KL_017", "date": "2025-12-25", "holiday_name": "Christmas", "state": "KL", "year": 2025, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.34/2024/GAD", "source_url": "https://kerala.gov.in", "notes": "Christmas Day"},

    # National Holidays 2025
    {"holiday_id": "H_2025_NAT_001", "date": "2025-01-26", "holiday_name": "Republic Day", "state": "ALL", "year": 2025, "is_gazetted": True, "is_national": True, "verification_status": "verified_official_gazette", "source_name": "Ministry of Personnel, Govt of India", "source_url": "https://dopt.gov.in", "notes": "National holiday"},
    {"holiday_id": "H_2025_NAT_002", "date": "2025-08-15", "holiday_name": "Independence Day", "state": "ALL", "year": 2025, "is_gazetted": True, "is_national": True, "verification_status": "verified_official_gazette", "source_name": "Ministry of Personnel, Govt of India", "source_url": "https://dopt.gov.in", "notes": "National holiday"},
    {"holiday_id": "H_2025_NAT_003", "date": "2025-10-02", "holiday_name": "Mahatma Gandhi Jayanti", "state": "ALL", "year": 2025, "is_gazetted": True, "is_national": True, "verification_status": "verified_official_gazette", "source_name": "Ministry of Personnel, Govt of India", "source_url": "https://dopt.gov.in", "notes": "National holiday"},

    # =========================================================================
    # YEAR 2026
    # =========================================================================
    # Tamil Nadu 2026 (G.O. (Ms.) No. 708, Public (Misc) Dept, 11.11.2025)
    {"holiday_id": "H_2026_TN_001", "date": "2026-01-01", "holiday_name": "New Year's Day", "state": "TN", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.708", "source_url": "https://www.tn.gov.in", "notes": "Public holiday under NI Act"},
    {"holiday_id": "H_2026_TN_002", "date": "2026-01-15", "holiday_name": "Pongal", "state": "TN", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.708", "source_url": "https://www.tn.gov.in", "notes": "Thai Pongal"},
    {"holiday_id": "H_2026_TN_003", "date": "2026-01-16", "holiday_name": "Thiruvalluvar Day", "state": "TN", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.708", "source_url": "https://www.tn.gov.in", "notes": "Pongal holiday sequence"},
    {"holiday_id": "H_2026_TN_004", "date": "2026-01-17", "holiday_name": "Uzhavar Thirunal", "state": "TN", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.708", "source_url": "https://www.tn.gov.in", "notes": "Farmers festival"},
    {"holiday_id": "H_2026_TN_005", "date": "2026-03-19", "holiday_name": "Telugu New Year's Day", "state": "TN", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.708", "source_url": "https://www.tn.gov.in", "notes": "Ugadi celebration"},
    {"holiday_id": "H_2026_TN_006", "date": "2026-03-21", "holiday_name": "Ramzan (Idul Fitr)", "state": "TN", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.708", "source_url": "https://www.tn.gov.in", "notes": "Subject to moon sighting"},
    {"holiday_id": "H_2026_TN_007", "date": "2026-03-31", "holiday_name": "Mahaveer Jayanthi", "state": "TN", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.708", "source_url": "https://www.tn.gov.in", "notes": "Jain festival"},
    {"holiday_id": "H_2026_TN_008", "date": "2026-04-03", "holiday_name": "Good Friday", "state": "TN", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.708", "source_url": "https://www.tn.gov.in", "notes": "Good Friday"},
    {"holiday_id": "H_2026_TN_009", "date": "2026-04-14", "holiday_name": "Tamil New Year / Dr Ambedkar Birthday", "state": "TN", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.708", "source_url": "https://www.tn.gov.in", "notes": "Tamil New Year Day"},
    {"holiday_id": "H_2026_TN_010", "date": "2026-05-01", "holiday_name": "May Day", "state": "TN", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.708", "source_url": "https://www.tn.gov.in", "notes": "Labour Day"},
    {"holiday_id": "H_2026_TN_011", "date": "2026-05-28", "holiday_name": "Bakrid (Eid al-Adha)", "state": "TN", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.708", "source_url": "https://www.tn.gov.in", "notes": "Islamic feast day"},
    {"holiday_id": "H_2026_TN_012", "date": "2026-06-26", "holiday_name": "Muharram", "state": "TN", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.708", "source_url": "https://www.tn.gov.in", "notes": "Ashura observance"},
    {"holiday_id": "H_2026_TN_013", "date": "2026-08-26", "holiday_name": "Milad-un-Nabi", "state": "TN", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.708", "source_url": "https://www.tn.gov.in", "notes": "Prophet Muhammad Birthday"},
    {"holiday_id": "H_2026_TN_014", "date": "2026-09-04", "holiday_name": "Krishna Jayanthi", "state": "TN", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.708", "source_url": "https://www.tn.gov.in", "notes": "Janmashtami"},
    {"holiday_id": "H_2026_TN_015", "date": "2026-09-14", "holiday_name": "Vinayagar Chathurthi", "state": "TN", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.708", "source_url": "https://www.tn.gov.in", "notes": "Ganesh Chaturthi"},
    {"holiday_id": "H_2026_TN_016", "date": "2026-10-19", "holiday_name": "Ayudha Pooja", "state": "TN", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.708", "source_url": "https://www.tn.gov.in", "notes": "Navaratri holiday"},
    {"holiday_id": "H_2026_TN_017", "date": "2026-10-20", "holiday_name": "Vijaya Dashami", "state": "TN", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.708", "source_url": "https://www.tn.gov.in", "notes": "Dussehra holiday"},
    {"holiday_id": "H_2026_TN_018", "date": "2026-12-25", "holiday_name": "Christmas Day", "state": "TN", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Tamil Nadu G.O.(Ms.) No.708", "source_url": "https://www.tn.gov.in", "notes": "Christmas"},

    # Karnataka 2026 (Notification No. DPAR 16 HHL 2025, 17.11.2025)
    {"holiday_id": "H_2026_KA_001", "date": "2026-01-15", "holiday_name": "Makara Sankranti", "state": "KA", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2025", "source_url": "https://dpar.karnataka.gov.in", "notes": "Uttarayana Punyakala"},
    {"holiday_id": "H_2026_KA_002", "date": "2026-03-19", "holiday_name": "Ugadi Festival", "state": "KA", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2025", "source_url": "https://dpar.karnataka.gov.in", "notes": "Kannada New Year"},
    {"holiday_id": "H_2026_KA_003", "date": "2026-03-21", "holiday_name": "Khutub-E-Ramzan", "state": "KA", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2025", "source_url": "https://dpar.karnataka.gov.in", "notes": "Id-ul-Fitr"},
    {"holiday_id": "H_2026_KA_004", "date": "2026-03-31", "holiday_name": "Mahaveera Jayanthi", "state": "KA", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2025", "source_url": "https://dpar.karnataka.gov.in", "notes": "Jain observance"},
    {"holiday_id": "H_2026_KA_005", "date": "2026-04-03", "holiday_name": "Good Friday", "state": "KA", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2025", "source_url": "https://dpar.karnataka.gov.in", "notes": "Christian observance"},
    {"holiday_id": "H_2026_KA_006", "date": "2026-04-14", "holiday_name": "Dr. B.R. Ambedkar Jayanthi", "state": "KA", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2025", "source_url": "https://dpar.karnataka.gov.in", "notes": "Ambedkar Birthday"},
    {"holiday_id": "H_2026_KA_007", "date": "2026-04-20", "holiday_name": "Basava Jayanthi / Akshaya Tritiya", "state": "KA", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2025", "source_url": "https://dpar.karnataka.gov.in", "notes": "Basavanna Birthday"},
    {"holiday_id": "H_2026_KA_008", "date": "2026-05-01", "holiday_name": "May Day", "state": "KA", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2025", "source_url": "https://dpar.karnataka.gov.in", "notes": "Labour Day"},
    {"holiday_id": "H_2026_KA_009", "date": "2026-05-28", "holiday_name": "Bakrid (Eid al-Adha)", "state": "KA", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2025", "source_url": "https://dpar.karnataka.gov.in", "notes": "Id-ul-Adha"},
    {"holiday_id": "H_2026_KA_010", "date": "2026-06-26", "holiday_name": "Last Day of Moharram", "state": "KA", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2025", "source_url": "https://dpar.karnataka.gov.in", "notes": "Ashura day"},
    {"holiday_id": "H_2026_KA_011", "date": "2026-08-26", "holiday_name": "Eid-Milad", "state": "KA", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2025", "source_url": "https://dpar.karnataka.gov.in", "notes": "Milad-un-Nabi"},
    {"holiday_id": "H_2026_KA_012", "date": "2026-09-14", "holiday_name": "Varasiddhi Vinayaka Vrata", "state": "KA", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2025", "source_url": "https://dpar.karnataka.gov.in", "notes": "Ganesh Chaturthi"},
    {"holiday_id": "H_2026_KA_013", "date": "2026-10-20", "holiday_name": "Mahanavami / Ayudhapooja", "state": "KA", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2025", "source_url": "https://dpar.karnataka.gov.in", "notes": "Dasara festival"},
    {"holiday_id": "H_2026_KA_014", "date": "2026-10-21", "holiday_name": "Vijayadasami", "state": "KA", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2025", "source_url": "https://dpar.karnataka.gov.in", "notes": "Dussehra Vijayadashami"},
    {"holiday_id": "H_2026_KA_015", "date": "2026-11-10", "holiday_name": "Balipadyami / Deepavali", "state": "KA", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2025", "source_url": "https://dpar.karnataka.gov.in", "notes": "Deepavali festival"},
    {"holiday_id": "H_2026_KA_016", "date": "2026-11-27", "holiday_name": "Kanakadasa Jayanthi", "state": "KA", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2025", "source_url": "https://dpar.karnataka.gov.in", "notes": "Kanakadasa Birthday"},
    {"holiday_id": "H_2026_KA_017", "date": "2026-12-25", "holiday_name": "Christmas", "state": "KA", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Karnataka DPAR 16 HHL 2025", "source_url": "https://dpar.karnataka.gov.in", "notes": "Christmas Day"},

    # ---------------------------------------------------------------------------
    # Kerala 2024 — supplementary entries (Cultural Observances & corrections)
    # ---------------------------------------------------------------------------
    # Thiruvonam 2024: Kerala Government gazettes Thiruvonam as an official
    # public holiday every year. The Thiruvonam nakshatra falls on 2024-09-15.
    # Source: KL G.O.(P) No.24/2023/GAD (same gazette as other KL 2024 holidays).
    {"holiday_id": "H_2024_KL_014", "date": "2024-09-15", "holiday_name": "Thiruvonam (Onam)", "state": "KL", "year": 2024, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.24/2023/GAD", "source_url": "https://kerala.gov.in", "notes": "Main Onam day; Thiruvonam nakshatra in Chingam 2024"},

    # Cultural observance: surrounding Onam festival days. These are NOT
    # official gazetted public holidays but are widely observed; labelled as
    # cultural_observance_assumption for modelling purposes.
    {"holiday_id": "H_2024_KL_COL_001", "date": "2024-09-14", "holiday_name": "Uthradam (Second Onam) — Cultural Observance", "state": "KL", "year": 2024, "is_gazetted": False, "is_national": False, "verification_status": "cultural_observance_assumption", "source_name": "Kerala tourism/cultural calendar — synthetic assumption", "source_url": "https://kerala.gov.in", "notes": "Cultural observance only; not a gazetted public holiday; synthetic scenario assumption for demand modelling"},
    {"holiday_id": "H_2024_KL_COL_002", "date": "2024-09-16", "holiday_name": "Avittom (Third Onam) — Cultural Observance", "state": "KL", "year": 2024, "is_gazetted": False, "is_national": False, "verification_status": "cultural_observance_assumption", "source_name": "Kerala tourism/cultural calendar — synthetic assumption", "source_url": "https://kerala.gov.in", "notes": "Cultural observance only; not a gazetted public holiday; synthetic scenario assumption for demand modelling"},

    # Vishu 2024: Vishu (Malayalam New Year) falls on April 14, 2024.
    # The KL 2024 gazette (G.O.(P) No.24/2023/GAD) did not separately list
    # Vishu as a gazetted public holiday for 2024 (it was gazetted in 2025 and
    # 2026). Added here as a cultural observance assumption.
    {"holiday_id": "H_2024_KL_COL_003", "date": "2024-04-14", "holiday_name": "Vishu (Malayalam New Year) — Cultural Observance", "state": "KL", "year": 2024, "is_gazetted": False, "is_national": False, "verification_status": "cultural_observance_assumption", "source_name": "Kerala cultural calendar — synthetic assumption", "source_url": "https://kerala.gov.in", "notes": "Vishu not separately gazetted in KL 2024; added as cultural_observance_assumption for demand modelling; profile_type=synthetic_scenario_assumption"},

    # ---------------------------------------------------------------------------
    # Kerala 2026 — supplementary cultural observance for Onam flanking day
    # ---------------------------------------------------------------------------
    # KL 2026 gazette already includes First Onam (Aug 25) and Thiruvonam (Aug 26).
    # Avittom (Third Onam, Aug 27) is not separately gazetted; added as cultural.
    {"holiday_id": "H_2026_KL_COL_001", "date": "2026-08-27", "holiday_name": "Avittom (Third Onam) — Cultural Observance", "state": "KL", "year": 2026, "is_gazetted": False, "is_national": False, "verification_status": "cultural_observance_assumption", "source_name": "Kerala tourism/cultural calendar — synthetic assumption", "source_url": "https://kerala.gov.in", "notes": "Cultural observance only; not a gazetted public holiday; synthetic scenario assumption for demand modelling"},

    # Kerala 2026 (G.O. (P) No. 14/2025/GAD, 31.10.2025)
    {"holiday_id": "H_2026_KL_001", "date": "2026-01-02", "holiday_name": "Mannam Jayanthi", "state": "KL", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.14/2025/GAD", "source_url": "https://kerala.gov.in", "notes": "Mannam Jayanthi"},
    {"holiday_id": "H_2026_KL_002", "date": "2026-03-20", "holiday_name": "Id-ul-Fitr (Ramzan)", "state": "KL", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.14/2025/GAD", "source_url": "https://kerala.gov.in", "notes": "Subject to moon sighting"},
    {"holiday_id": "H_2026_KL_003", "date": "2026-04-02", "holiday_name": "Maundy Thursday", "state": "KL", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.14/2025/GAD", "source_url": "https://kerala.gov.in", "notes": "Holy Thursday"},
    {"holiday_id": "H_2026_KL_004", "date": "2026-04-03", "holiday_name": "Good Friday", "state": "KL", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.14/2025/GAD", "source_url": "https://kerala.gov.in", "notes": "Good Friday"},
    {"holiday_id": "H_2026_KL_005", "date": "2026-04-14", "holiday_name": "Dr. B.R. Ambedkar Jayanthi", "state": "KL", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.14/2025/GAD", "source_url": "https://kerala.gov.in", "notes": "Ambedkar Birthday"},
    {"holiday_id": "H_2026_KL_006", "date": "2026-04-15", "holiday_name": "Vishu", "state": "KL", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.14/2025/GAD", "source_url": "https://kerala.gov.in", "notes": "Malayalam New Year harvest day"},
    {"holiday_id": "H_2026_KL_007", "date": "2026-05-01", "holiday_name": "May Day", "state": "KL", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.14/2025/GAD", "source_url": "https://kerala.gov.in", "notes": "May Day"},
    {"holiday_id": "H_2026_KL_008", "date": "2026-05-27", "holiday_name": "Id-ul-Ad'ha (Bakrid)", "state": "KL", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.14/2025/GAD", "source_url": "https://kerala.gov.in", "notes": "Bakrid feast"},
    {"holiday_id": "H_2026_KL_009", "date": "2026-06-25", "holiday_name": "Muharram", "state": "KL", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.14/2025/GAD", "source_url": "https://kerala.gov.in", "notes": "Muharram"},
    {"holiday_id": "H_2026_KL_010", "date": "2026-08-25", "holiday_name": "First Onam / Milad-i-Sherif", "state": "KL", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.14/2025/GAD", "source_url": "https://kerala.gov.in", "notes": "Onam day 1"},
    {"holiday_id": "H_2026_KL_011", "date": "2026-08-26", "holiday_name": "Thiruvonam", "state": "KL", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.14/2025/GAD", "source_url": "https://kerala.gov.in", "notes": "Thiruvonam core day"},
    {"holiday_id": "H_2026_KL_012", "date": "2026-08-28", "holiday_name": "Sree Narayana Guru Jayanthi / Ayyankali Jayanthi", "state": "KL", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.14/2025/GAD", "source_url": "https://kerala.gov.in", "notes": "Guru Jayanthi"},
    {"holiday_id": "H_2026_KL_013", "date": "2026-09-04", "holiday_name": "Sreekrishna Jayanthi", "state": "KL", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.14/2025/GAD", "source_url": "https://kerala.gov.in", "notes": "Janmashtami"},
    {"holiday_id": "H_2026_KL_014", "date": "2026-09-20", "holiday_name": "Sree Narayana Guru Samadhi Day", "state": "KL", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.14/2025/GAD", "source_url": "https://kerala.gov.in", "notes": "Guru Samadhi"},
    {"holiday_id": "H_2026_KL_015", "date": "2026-10-19", "holiday_name": "Mahanavami", "state": "KL", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.14/2025/GAD", "source_url": "https://kerala.gov.in", "notes": "Puja holiday"},
    {"holiday_id": "H_2026_KL_016", "date": "2026-10-20", "holiday_name": "Vijayadasami", "state": "KL", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.14/2025/GAD", "source_url": "https://kerala.gov.in", "notes": "Vidyarambham"},
    {"holiday_id": "H_2026_KL_017", "date": "2026-12-25", "holiday_name": "Christmas", "state": "KL", "year": 2026, "is_gazetted": True, "is_national": False, "verification_status": "verified_official_gazette", "source_name": "Govt of Kerala G.O.(P) No.14/2025/GAD", "source_url": "https://kerala.gov.in", "notes": "Christmas Day"},

    # National Holidays 2026
    {"holiday_id": "H_2026_NAT_001", "date": "2026-01-26", "holiday_name": "Republic Day", "state": "ALL", "year": 2026, "is_gazetted": True, "is_national": True, "verification_status": "verified_official_gazette", "source_name": "Ministry of Personnel, Govt of India", "source_url": "https://dopt.gov.in", "notes": "National holiday"},
    {"holiday_id": "H_2026_NAT_002", "date": "2026-08-15", "holiday_name": "Independence Day", "state": "ALL", "year": 2026, "is_gazetted": True, "is_national": True, "verification_status": "verified_official_gazette", "source_name": "Ministry of Personnel, Govt of India", "source_url": "https://dopt.gov.in", "notes": "National holiday"},
    {"holiday_id": "H_2026_NAT_003", "date": "2026-10-02", "holiday_name": "Mahatma Gandhi Jayanti", "state": "ALL", "year": 2026, "is_gazetted": True, "is_national": True, "verification_status": "verified_official_gazette", "source_name": "Ministry of Personnel, Govt of India", "source_url": "https://dopt.gov.in", "notes": "National holiday"},
]

HOLIDAY_EVENT_COLUMNS = [
    "holiday_id",
    "date",
    "holiday_name",
    "state",
    "year",
    "is_gazetted",
    "is_national",
    "verification_status",
    "source_name",
    "source_url",
    "notes",
]

CALENDAR_FEATURE_COLUMNS = [
    "date",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_name",
    "week_of_year",
    "quarter",
    "is_weekend",
    "is_monday",
    "is_friday",
    "is_month_start",
    "is_month_end",
    "season",
    "is_holiday_ka",
    "is_holiday_kl",
    "is_holiday_tn",
    "is_national_holiday",
    "is_pre_holiday",
    "is_post_holiday",
    "is_long_weekend",
    "is_school_vacation_ka",
    "is_school_vacation_kl",
    "is_school_vacation_tn",
    "is_college_vacation_ka",
    "is_college_vacation_kl",
    "is_college_vacation_tn",
]


# ---------------------------------------------------------------------------
# Vacation & Seasonal Calculation Logic
# ---------------------------------------------------------------------------
def get_meteorological_season(month: int) -> str:
    """Return the standard Indian Meteorological Department (IMD) season.

    Args:
        month: Calendar month (1 to 12).

    Returns:
        One of 'Winter', 'Summer', 'Monsoon', 'Post-Monsoon'.
    """
    if month in (1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Summer"
    elif month in (6, 7, 8, 9):
        return "Monsoon"
    else:  # 10, 11, 12
        return "Post-Monsoon"


def is_school_vacation_karnataka(d: date) -> bool:
    """Documented synthetic scenario assumption for Karnataka school vacations.

    - Summer Break: April 11 to May 28
    - Dasara / Midterm Break: October 1 to October 14
    """
    m, day = d.month, d.day
    # Summer break
    if (m == 4 and day >= 11) or (m == 5 and day <= 28):
        return True
    # Dasara break
    if m == 10 and 1 <= day <= 14:
        return True
    return False


def is_school_vacation_kerala(d: date) -> bool:
    """Documented synthetic scenario assumption for Kerala school vacations.

    - Summer Break: April 1 to May 31 (Official General Education Dept rule)
    - Onam Break: ~10 days in late August or early September
    - Christmas Break: December 23 to January 2
    """
    m, day = d.month, d.day
    # Summer vacation
    if m in (4, 5):
        return True
    # Onam vacation
    if d.year == 2024 and (m == 9 and 10 <= day <= 20):
        return True
    if d.year == 2025 and (m == 8 and 30 <= day <= 31 or m == 9 and 1 <= day <= 8):
        return True
    if d.year == 2026 and (m == 8 and 21 <= day <= 31):
        return True
    # Christmas break
    if (m == 12 and day >= 23) or (m == 1 and day <= 2):
        return True
    return False


def is_school_vacation_tamil_nadu(d: date) -> bool:
    """Documented synthetic scenario assumption for Tamil Nadu school vacations.

    - Summer Break: April 21 to June 5
    - Pongal Holidays: January 13 to January 18
    """
    m, day = d.month, d.day
    # Summer vacation
    if (m == 4 and day >= 21) or m == 5 or (m == 6 and day <= 5):
        return True
    # Pongal holiday window
    if m == 1 and 13 <= day <= 18:
        return True
    return False


def is_college_vacation_karnataka(d: date) -> bool:
    """Documented synthetic scenario assumption for Karnataka higher education.

    Synthetic assumption — not an official government mandate:
    - Summer Break: May 15 to July 10 (post-exam semester recess)
    - Winter Break: December 22 to January 3 (inter-semester holiday)
    """
    m, day = d.month, d.day
    if (m == 5 and day >= 15) or m == 6 or (m == 7 and day <= 10):
        return True
    if (m == 12 and day >= 22) or (m == 1 and day <= 3):
        return True
    return False


def is_college_vacation_kerala(d: date) -> bool:
    """Documented synthetic scenario assumption for Kerala higher education.

    Synthetic assumption — not an official government mandate:
    - Summer Break: May 1 to July 15 (longer break reflecting Kerala academic calendar)
    - Christmas Break: December 23 to January 3
    """
    m, day = d.month, d.day
    if m == 5 or m == 6 or (m == 7 and day <= 15):
        return True
    if (m == 12 and day >= 23) or (m == 1 and day <= 3):
        return True
    return False


def is_college_vacation_tamil_nadu(d: date) -> bool:
    """Documented synthetic scenario assumption for Tamil Nadu higher education.

    Synthetic assumption — not an official government mandate:
    - Summer Break: May 15 to July 10
    - Pongal Window: January 13 to January 18
    - Winter Break: December 22 to January 3
    """
    m, day = d.month, d.day
    if (m == 5 and day >= 15) or m == 6 or (m == 7 and day <= 10):
        return True
    if m == 1 and 13 <= day <= 18:
        return True
    if (m == 12 and day >= 22) or (m == 1 and day <= 3):
        return True
    return False


# ---------------------------------------------------------------------------
# Core Feature Generator Engine
# ---------------------------------------------------------------------------
def build_holiday_events() -> pd.DataFrame:
    """Build, validate, and return the official holiday events table."""
    df = pd.DataFrame(OFFICIAL_HOLIDAY_EVENTS)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Validate schema
    missing = set(HOLIDAY_EVENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing holiday event columns: {missing}")

    if df["holiday_id"].duplicated().any():
        dups = df.loc[df["holiday_id"].duplicated(), "holiday_id"].tolist()
        raise ValueError(f"Duplicate holiday IDs: {dups}")

    return df[HOLIDAY_EVENT_COLUMNS].sort_values(by=["date", "state"]).reset_index(drop=True)


def generate_calendar_features(
    start_date: str = SIMULATION_START_DATE,
    end_date: str = SIMULATION_END_DATE,
    holiday_events: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the 1,096-day calendar features and holiday event datasets.

    Args:
        start_date: Start date string (YYYY-MM-DD), default '2024-01-01'.
        end_date: End date string (YYYY-MM-DD), default '2026-12-31'.
        holiday_events: Optional custom holiday events DataFrame.

    Returns:
        Tuple of (calendar_features_df, holiday_events_df).
    """
    if holiday_events is None:
        holiday_events = build_holiday_events()

    # Generate continuous date range
    dt_range = pd.date_range(start=start_date, end=end_date, freq="D")
    expected_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1

    if len(dt_range) != expected_days:
        raise ValueError(
            f"Expected {expected_days} dates between {start_date} and {end_date}, "
            f"got {len(dt_range)}"
        )

    # Pre-index holiday sets for rapid lookup.
    # IMPORTANT: only gazetted official public holidays contribute to the
    # is_holiday_* flags. Cultural observance entries (is_gazetted=False)
    # are stored in holiday_events.csv for reference but must NOT pollute
    # the official holiday indicators.
    gazetted = holiday_events[holiday_events["is_gazetted"] == True]
    ka_holidays: Set[str] = set(
        gazetted[gazetted["state"].isin(["KA", "ALL"])]["date"]
    )
    kl_holidays: Set[str] = set(
        gazetted[gazetted["state"].isin(["KL", "ALL"])]["date"]
    )
    tn_holidays: Set[str] = set(
        gazetted[gazetted["state"].isin(["TN", "ALL"])]["date"]
    )
    national_holidays: Set[str] = set(
        gazetted[gazetted["is_national"] == True]["date"]
    )
    all_public_holidays: Set[str] = ka_holidays | kl_holidays | tn_holidays

    rows = []
    for cur_dt in dt_range:
        d = cur_dt.date()
        date_str = cur_dt.strftime("%Y-%m-%d")

        # Basic calendar dimensions
        year = d.year
        month = d.month
        day = d.day
        day_of_week = cur_dt.day_of_week  # 0=Monday, 6=Sunday
        day_name = cur_dt.day_name()
        week_of_year = int(cur_dt.isocalendar().week)
        quarter = cur_dt.quarter

        is_weekend = day_of_week in (5, 6)
        is_monday = day_of_week == 0
        is_friday = day_of_week == 4
        is_month_start = cur_dt.is_month_start
        is_month_end = cur_dt.is_month_end
        season = get_meteorological_season(month)

        # Holiday flags
        is_h_ka = date_str in ka_holidays
        is_h_kl = date_str in kl_holidays
        is_h_tn = date_str in tn_holidays
        is_nat = date_str in national_holidays
        is_any_h = is_h_ka or is_h_kl or is_h_tn

        # Pre/post holiday calculations
        prev_date_str = (cur_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        next_date_str = (cur_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        is_pre_holiday = next_date_str in all_public_holidays
        is_post_holiday = prev_date_str in all_public_holidays

        # Long weekend: If holiday is on Friday, Monday, or creates a 3+ day block with weekend
        is_long_weekend = False
        if is_any_h:
            if is_friday or is_monday:
                is_long_weekend = True
            elif is_weekend:
                # Holiday falls on weekend; check if Friday or Monday is also off
                if prev_date_str in all_public_holidays or next_date_str in all_public_holidays:
                    is_long_weekend = True
        elif is_weekend:
            # Check adjacent Friday or Monday holiday
            if is_friday and is_pre_holiday:
                is_long_weekend = True
            elif day_of_week == 5:  # Saturday
                # Friday before or Monday after is holiday
                fri_str = (cur_dt - timedelta(days=1)).strftime("%Y-%m-%d")
                mon_str = (cur_dt + timedelta(days=2)).strftime("%Y-%m-%d")
                if fri_str in all_public_holidays or mon_str in all_public_holidays:
                    is_long_weekend = True
            elif day_of_week == 6:  # Sunday
                fri_str = (cur_dt - timedelta(days=2)).strftime("%Y-%m-%d")
                mon_str = (cur_dt + timedelta(days=1)).strftime("%Y-%m-%d")
                if fri_str in all_public_holidays or mon_str in all_public_holidays:
                    is_long_weekend = True

        # Vacation assumptions
        vac_ka = is_school_vacation_karnataka(d)
        vac_kl = is_school_vacation_kerala(d)
        vac_tn = is_school_vacation_tamil_nadu(d)
        vac_col_ka = is_college_vacation_karnataka(d)
        vac_col_kl = is_college_vacation_kerala(d)
        vac_col_tn = is_college_vacation_tamil_nadu(d)

        rows.append({
            "date": date_str,
            "year": year,
            "month": month,
            "day": day,
            "day_of_week": day_of_week,
            "day_name": day_name,
            "week_of_year": week_of_year,
            "quarter": quarter,
            "is_weekend": is_weekend,
            "is_monday": is_monday,
            "is_friday": is_friday,
            "is_month_start": is_month_start,
            "is_month_end": is_month_end,
            "season": season,
            "is_holiday_ka": is_h_ka,
            "is_holiday_kl": is_h_kl,
            "is_holiday_tn": is_h_tn,
            "is_national_holiday": is_nat,
            "is_pre_holiday": is_pre_holiday,
            "is_post_holiday": is_post_holiday,
            "is_long_weekend": is_long_weekend,
            "is_school_vacation_ka": vac_ka,
            "is_school_vacation_kl": vac_kl,
            "is_school_vacation_tn": vac_tn,
            "is_college_vacation_ka": vac_col_ka,
            "is_college_vacation_kl": vac_col_kl,
            "is_college_vacation_tn": vac_col_tn,
        })

    cal_df = pd.DataFrame(rows)
    validate_calendar_features(cal_df)
    return cal_df, holiday_events


def validate_calendar_features(df: pd.DataFrame) -> None:
    """Validate calendar feature table integrity, length, and column schemas.

    Raises:
        ValueError: If dates are discontinuous, rows != 1096, columns are missing,
            or values violate logical constraints.
    """
    # 1. Required columns
    missing = set(CALENDAR_FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required calendar columns: {sorted(missing)}")

    # 2. Exactly 1,096 days
    if len(df) != 1096:
        raise ValueError(f"Expected exactly 1,096 dates, found {len(df)}")

    # 3. Continuous date sequence
    dates = pd.to_datetime(df["date"])
    if not dates.is_monotonic_increasing:
        raise ValueError("Dates in calendar table must be strictly monotonically increasing.")

    day_diffs = dates.diff().dropna()
    if not (day_diffs == pd.Timedelta(days=1)).all():
        raise ValueError("Calendar dates contain gaps or duplicates.")

    # 4. No null values
    if df[CALENDAR_FEATURE_COLUMNS].isna().any().any():
        null_counts = df[CALENDAR_FEATURE_COLUMNS].isna().sum()
        raise ValueError(f"Null values detected in calendar features: {null_counts[null_counts > 0].to_dict()}")

    # 5. Logical constraints
    if not df["year"].isin([2024, 2025, 2026]).all():
        raise ValueError("Year must be within 2024-2026.")

    if not df["month"].between(1, 12).all():
        raise ValueError("Month must be between 1 and 12.")

    if not df["day_of_week"].between(0, 6).all():
        raise ValueError("day_of_week must be between 0 (Monday) and 6 (Sunday).")

    if not df["quarter"].between(1, 4).all():
        raise ValueError("Quarter must be between 1 and 4.")

    valid_seasons = {"Winter", "Summer", "Monsoon", "Post-Monsoon"}
    invalid_seasons = set(df["season"].unique()) - valid_seasons
    if invalid_seasons:
        raise ValueError(f"Invalid seasons detected: {invalid_seasons}")

    # National holidays must be exactly on 01-26, 08-15, 10-02
    national_dates = set(df[df["is_national_holiday"]]["date"].str[5:])
    expected_national_md = {"01-26", "08-15", "10-02"}
    if national_dates != expected_national_md:
        raise ValueError(f"National holiday month-days mismatch. Expected {expected_national_md}, got {national_dates}")


def main() -> None:
    """Generate, validate, and write calendar features and holiday events CSVs."""
    output_cal_path = RAW_DATA_DIR / "calendar_features.csv"
    output_hol_path = RAW_DATA_DIR / "holiday_events.csv"

    output_cal_path.parent.mkdir(parents=True, exist_ok=True)

    cal_df, hol_df = generate_calendar_features()

    cal_df.to_csv(output_cal_path, index=False)
    hol_df.to_csv(output_hol_path, index=False)

    print(f"\nCreated {len(cal_df)} calendar dates: {output_cal_path.resolve()}")
    print(f"Created {len(hol_df)} official holiday events: {output_hol_path.resolve()}")
    print("=" * 105)
    print("BUS DEMAND AI - CALENDAR AND SEASONAL FEATURES SUMMARY")
    print("=" * 105)
    print(f"Total Dates: {len(cal_df)} ({cal_df['date'].min()} to {cal_df['date'].max()})")
    print(f"Weekend Days: {cal_df['is_weekend'].sum()}")
    print(f"Karnataka Holidays: {cal_df['is_holiday_ka'].sum()}")
    print(f"Kerala Holidays: {cal_df['is_holiday_kl'].sum()}")
    print(f"Tamil Nadu Holidays: {cal_df['is_holiday_tn'].sum()}")
    print(f"Long Weekends: {cal_df['is_long_weekend'].sum()}")
    print(f"College Vacation (KA): {cal_df['is_college_vacation_ka'].sum()} days")
    print(f"College Vacation (KL): {cal_df['is_college_vacation_kl'].sum()} days")
    print(f"College Vacation (TN): {cal_df['is_college_vacation_tn'].sum()} days")
    print("\nSample Calendar Records:")
    preview_cols = [
        "date", "day_name", "season", "is_weekend",
        "is_holiday_ka", "is_holiday_kl", "is_holiday_tn",
        "is_long_weekend"
    ]
    print(cal_df[preview_cols].head(10).to_string(index=False))
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
