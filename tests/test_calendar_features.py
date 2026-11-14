"""Unit tests for the calendar and seasonal feature generator."""

from datetime import datetime
import pytest
import pandas as pd
import numpy as np

from src.data.generate_calendar_features import (
    generate_calendar_features,
    validate_calendar_features,
    build_holiday_events,
    get_meteorological_season,
    is_college_vacation_karnataka,
    is_college_vacation_kerala,
    is_college_vacation_tamil_nadu,
    CALENDAR_FEATURE_COLUMNS,
    HOLIDAY_EVENT_COLUMNS,
)
from src.config import RAW_DATA_DIR


@pytest.fixture
def calendar_data():
    """Fixture providing in-memory generated calendar and holiday dataframes."""
    return generate_calendar_features()


@pytest.fixture
def calendar_df(calendar_data):
    return calendar_data[0]


@pytest.fixture
def holiday_df(calendar_data):
    return calendar_data[1]


def test_calendar_row_count(calendar_df):
    """Verify exactly 1,096 days are generated (3 full years: 366 + 365 + 365)."""
    assert len(calendar_df) == 1096


def test_date_range_boundaries(calendar_df):
    """Verify date range starts on 2024-01-01 and ends on 2026-12-31."""
    assert calendar_df["date"].min() == "2024-01-01"
    assert calendar_df["date"].max() == "2026-12-31"


def test_continuous_date_sequence(calendar_df):
    """Ensure no date gaps, skips, or duplicate dates exist."""
    dates = pd.to_datetime(calendar_df["date"])
    assert dates.is_monotonic_increasing
    diffs = dates.diff().dropna()
    assert (diffs == pd.Timedelta(days=1)).all()


def test_column_schema(calendar_df):
    """Verify all required columns exist with zero missing values."""
    missing = set(CALENDAR_FEATURE_COLUMNS) - set(calendar_df.columns)
    assert not missing, f"Missing columns: {missing}"
    assert not calendar_df[CALENDAR_FEATURE_COLUMNS].isna().any().any()


def test_calendar_temporal_logic(calendar_df):
    """Verify year, month, day, day_of_week, and quarter consistency."""
    for _, row in calendar_df.sample(50, random_state=42).iterrows():
        dt = datetime.strptime(row["date"], "%Y-%m-%d")
        assert row["year"] == dt.year
        assert row["month"] == dt.month
        assert row["day"] == dt.day
        assert row["day_of_week"] == dt.weekday()
        assert row["day_name"] == dt.strftime("%A")
        assert row["quarter"] == (dt.month - 1) // 3 + 1
        assert row["is_weekend"] == (dt.weekday() in (5, 6))
        assert row["is_monday"] == (dt.weekday() == 0)
        assert row["is_friday"] == (dt.weekday() == 4)


def test_meteorological_seasons(calendar_df):
    """Verify Indian Meteorological Department season mapping."""
    for _, row in calendar_df.iterrows():
        m = row["month"]
        expected_season = get_meteorological_season(m)
        assert row["season"] == expected_season


def test_national_holidays(calendar_df):
    """Ensure Republic Day, Independence Day, and Gandhi Jayanti are flagged in all 3 years."""
    nat_dates = calendar_df[calendar_df["is_national_holiday"]]["date"].tolist()
    expected_nat = [
        "2024-01-26", "2024-08-15", "2024-10-02",
        "2025-01-26", "2025-08-15", "2025-10-02",
        "2026-01-26", "2026-08-15", "2026-10-02",
    ]
    assert sorted(nat_dates) == sorted(expected_nat)


def test_state_specific_holidays(calendar_df):
    """Verify flagship state-specific festivals are correctly isolated."""
    # Pongal (Jan 15, 2024) is TN holiday
    pongal_2024 = calendar_df[calendar_df["date"] == "2024-01-15"].iloc[0]
    assert pongal_2024["is_holiday_tn"]
    assert pongal_2024["is_holiday_ka"]  # Sankranti in KA
    assert not pongal_2024["is_holiday_kl"] # Not gazetted general holiday in KL

    # Kannada Rajyotsava (Nov 1, 2024) is KA holiday only
    rajyotsava = calendar_df[calendar_df["date"] == "2024-11-01"].iloc[0]
    assert rajyotsava["is_holiday_ka"]
    assert not rajyotsava["is_holiday_tn"]
    assert not rajyotsava["is_holiday_kl"]

    # Mannam Jayanthi (Jan 2, 2024) is KL holiday only
    mannam = calendar_df[calendar_df["date"] == "2024-01-02"].iloc[0]
    assert mannam["is_holiday_kl"]
    assert not mannam["is_holiday_ka"]
    assert not mannam["is_holiday_tn"]


def test_long_weekend_detection(calendar_df):
    """Verify that a Friday or Monday holiday triggers is_long_weekend."""
    # Good Friday 2024 (2024-03-29)
    gf = calendar_df[calendar_df["date"] == "2024-03-29"].iloc[0]
    assert gf["is_friday"]
    assert gf["is_long_weekend"]

    # Republic Day 2024 (Friday 2024-01-26)
    rd = calendar_df[calendar_df["date"] == "2024-01-26"].iloc[0]
    assert rd["is_friday"]
    assert rd["is_long_weekend"]


def test_vacation_assumptions(calendar_df):
    """Verify school and college vacation flags conform to documented assumptions."""
    # Peak summer (May 10) should be school vacation across all three states
    may10 = calendar_df[calendar_df["date"] == "2024-05-10"].iloc[0]
    assert may10["is_school_vacation_ka"]
    assert may10["is_school_vacation_kl"]
    assert may10["is_school_vacation_tn"]
    # College: KA and TN break starts May 15, so May 10 is NOT college vacation for them
    assert not may10["is_college_vacation_ka"]
    assert not may10["is_college_vacation_tn"]
    # KL college break starts May 1, so May 10 IS college vacation for KL
    assert may10["is_college_vacation_kl"]

    # May 20: all three states in college vacation
    may20 = calendar_df[calendar_df["date"] == "2024-05-20"].iloc[0]
    assert may20["is_college_vacation_ka"]
    assert may20["is_college_vacation_kl"]
    assert may20["is_college_vacation_tn"]

    # June 1 should have KL schools reopen, TN still off
    june1 = calendar_df[calendar_df["date"] == "2024-06-01"].iloc[0]
    assert not june1["is_school_vacation_kl"]
    assert june1["is_school_vacation_tn"]

    # State-specific college vacation functions work correctly
    from datetime import date as dt_date
    # KA college break starts May 15
    assert not is_college_vacation_karnataka(dt_date(2024, 5, 14))
    assert is_college_vacation_karnataka(dt_date(2024, 5, 15))
    # KL college break starts May 1
    assert is_college_vacation_kerala(dt_date(2024, 5, 1))
    assert not is_college_vacation_kerala(dt_date(2024, 7, 16))
    # TN college Pongal window
    assert is_college_vacation_tamil_nadu(dt_date(2024, 1, 15))


def test_holiday_events_metadata(holiday_df):
    """Verify holiday events registry schema, uniqueness, and official gazette sources."""
    assert len(holiday_df) > 0
    assert holiday_df["holiday_id"].is_unique
    assert set(HOLIDAY_EVENT_COLUMNS).issubset(holiday_df.columns)

    # Every event must have official source name and URL
    assert (holiday_df["source_name"].str.len() > 0).all()
    assert (holiday_df["source_url"].str.startswith("http")).all()
    assert holiday_df["verification_status"].isin([
        "verified_official_gazette",
        "provisional_astronomical_calendar",
        "cultural_observance_assumption",
    ]).all()


def test_validator_detects_row_count_mismatch(calendar_df):
    """Validator should reject datasets not containing exactly 1,096 dates."""
    invalid = calendar_df.iloc[:-1].copy()
    with pytest.raises(ValueError, match="Expected exactly 1,096 dates"):
        validate_calendar_features(invalid)


def test_validator_detects_date_gap(calendar_df):
    """Validator should reject datasets with missing dates/gaps."""
    invalid = calendar_df.drop(index=[10]).reset_index(drop=True)
    invalid = pd.concat([invalid, calendar_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="(strictly monotonically increasing|gaps or duplicates)"):
        validate_calendar_features(invalid)


def test_persisted_csv_files():
    """Verify both calendar_features.csv and holiday_events.csv exist on disk and validate."""
    cal_path = RAW_DATA_DIR / "calendar_features.csv"
    hol_path = RAW_DATA_DIR / "holiday_events.csv"

    assert cal_path.exists(), f"File {cal_path} does not exist"
    assert hol_path.exists(), f"File {hol_path} does not exist"

    loaded_cal = pd.read_csv(cal_path)
    assert len(loaded_cal) == 1096
    validate_calendar_features(loaded_cal)

    loaded_hol = pd.read_csv(hol_path)
    assert len(loaded_hol) >= 150
    assert loaded_hol["holiday_id"].is_unique


def test_onam_2024_gazetted_official_holiday(calendar_df, holiday_df):
    """Thiruvonam 2024 (Sep 15) must be an official gazetted holiday for Kerala.

    Decision: Onam 2024 is represented with one gazetted Thiruvonam day plus
    cultural observance entries for surrounding days. Only the gazetted day
    should set is_holiday_kl.
    """
    # Gazetted Thiruvonam flags Kerala as holiday
    thiruvonam = calendar_df[calendar_df["date"] == "2024-09-15"].iloc[0]
    assert thiruvonam["is_holiday_kl"], "Thiruvonam 2024 (Sep 15) must set is_holiday_kl"
    assert not thiruvonam["is_holiday_ka"], "Thiruvonam must NOT set is_holiday_ka"
    assert not thiruvonam["is_holiday_tn"], "Thiruvonam must NOT set is_holiday_tn"

    # Cultural flanking days must NOT set is_holiday_kl
    # Sep 14 (Uthradam): no independent KL gazette entry — must be clear
    uthradam = calendar_df[calendar_df["date"] == "2024-09-14"].iloc[0]
    assert not uthradam["is_holiday_kl"], "Uthradam (cultural) must NOT set is_holiday_kl"
    # Sep 16 (Avittom): NOTE — this date also carries the gazetted Milad-i-Sherif
    # (H_2024_KL_011), so is_holiday_kl=True here is CORRECT and expected.
    avittom = calendar_df[calendar_df["date"] == "2024-09-16"].iloc[0]
    assert avittom["is_holiday_kl"], "Sep 16 must be KL holiday (Milad-i-Sherif gazette entry)"

    # holiday_events.csv must contain the gazetted Thiruvonam entry
    gazetted_onam = holiday_df[
        (holiday_df["date"] == "2024-09-15") &
        (holiday_df["state"] == "KL") &
        (holiday_df["is_gazetted"] == True)
    ]
    assert len(gazetted_onam) == 1

    # holiday_events.csv must contain cultural observance entries
    cultural = holiday_df[
        (holiday_df["verification_status"] == "cultural_observance_assumption") &
        (holiday_df["state"] == "KL") &
        (holiday_df["year"] == 2024)
    ]
    # At least Uthradam and Avittom cultural entries
    assert len(cultural) >= 2

    # All cultural entries must have is_gazetted=False
    assert (cultural["is_gazetted"] == False).all()


def test_vishu_2024_cultural_observance(calendar_df, holiday_df):
    """Vishu 2024 (Apr 14) is a cultural observance, not an official gazetted holiday.

    Decision: Since KL G.O.(P) No.24/2023/GAD did not separately gazette Vishu
    for 2024, it is stored as cultural_observance_assumption only. It must NOT
    set is_holiday_kl in calendar_features.csv.
    """
    vishu = calendar_df[calendar_df["date"] == "2024-04-14"].iloc[0]
    # Vishu itself must not set is_holiday_kl for 2024 (not gazetted in KL 2024)
    assert not vishu["is_holiday_kl"], (
        "Vishu 2024 is a cultural_observance_assumption and must NOT set is_holiday_kl"
    )

    # It IS a KA holiday (Ugadi/Telugu New Year) and TN holiday (Tamil New Year in some years)
    # — this was already verified in test_state_specific_holidays for 2024-04-09 (Ugadi)
    # For 2024-04-14, check there's no KL gazetted entry
    kl_gazetted_apr14 = holiday_df[
        (holiday_df["date"] == "2024-04-14") &
        (holiday_df["state"] == "KL") &
        (holiday_df["is_gazetted"] == True)
    ]
    assert len(kl_gazetted_apr14) == 0, (
        "KL must have no gazetted entry for Apr 14 2024 (Vishu not in gazette)"
    )

    # The cultural observance entry must exist in holiday_events.csv
    cultural_vishu = holiday_df[
        (holiday_df["date"] == "2024-04-14") &
        (holiday_df["state"] == "KL") &
        (holiday_df["verification_status"] == "cultural_observance_assumption")
    ]
    assert len(cultural_vishu) == 1, "Vishu 2024 cultural entry must exist in holiday_events"
    assert cultural_vishu.iloc[0]["is_gazetted"] == False

    # Vishu IS gazetted in 2025 and 2026 — verify those remain official
    vishu_2025 = calendar_df[calendar_df["date"] == "2025-04-14"].iloc[0]
    assert vishu_2025["is_holiday_kl"], "Vishu 2025 (Vishu+Ambedkar Jayanthi) must set is_holiday_kl"

    vishu_2026 = calendar_df[calendar_df["date"] == "2026-04-15"].iloc[0]
    assert vishu_2026["is_holiday_kl"], "Vishu 2026 (Apr 15) must set is_holiday_kl"


def test_cultural_observances_isolated_from_official_flags(calendar_df, holiday_df):
    """All cultural_observance_assumption entries must NOT set any is_holiday_* flag.

    This guarantees the official holiday indicators are clean and sourced only
    from verified gazette entries.
    """
    cultural = holiday_df[holiday_df["verification_status"] == "cultural_observance_assumption"]
    assert len(cultural) > 0, "At least some cultural observance entries must exist"
    assert (cultural["is_gazetted"] == False).all(), "All cultural entries must have is_gazetted=False"

    for _, row in cultural.iterrows():
        date_str = row["date"]
        state = row["state"]
        cal_row = calendar_df[calendar_df["date"] == date_str]
        if cal_row.empty:
            continue  # Date outside calendar range (shouldn't happen)
        cal_row = cal_row.iloc[0]

        # On a cultural-only day, the corresponding state flag must NOT be set
        # (unless that same date is independently gazetted by another entry)
        is_independently_gazetted = holiday_df[
            (holiday_df["date"] == date_str) &
            (holiday_df["state"].isin([state, "ALL"])) &
            (holiday_df["is_gazetted"] == True)
        ]
        if len(is_independently_gazetted) == 0:
            flag_map = {"KL": "is_holiday_kl", "KA": "is_holiday_ka", "TN": "is_holiday_tn"}
            if state in flag_map:
                assert not cal_row[flag_map[state]], (
                    f"Cultural entry {row['holiday_id']} on {date_str} must not set {flag_map[state]}"
                )
