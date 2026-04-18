import pandas as pd
import numpy as np

# =========================================================
# ACCOR HOTEL RECOMMENDATION SYSTEM
# APP-READY VERSION
# =========================================================

# =========================
# 1. LOAD DATA
# =========================
user_requests = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Accor Project/user_requests.csv")
bookings = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Accor Project/user_bookings.csv")
hotels = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Accor Project/hotels.csv")

# =========================
# 2. CLEAN DATA
# =========================
for df in [bookings, hotels]:
    df.columns = df.columns.str.strip().str.lower()

for col in ["hotel_id", "brand", "tier", "address", "city", "country", "amenities"]:
    if col in hotels.columns:
        hotels[col] = hotels[col].astype(str).str.strip()

for col in ["user_id", "loyalty_tier", "hotel_id", "trip_purpose", "party_type"]:
    if col in bookings.columns:
        bookings[col] = bookings[col].astype(str).str.strip()

for col in ["price_range_low", "price_range_high"]:
    if col in hotels.columns:
        hotels[col] = pd.to_numeric(hotels[col], errors="coerce")

for col in ["length_of_stay", "price_paid", "rating"]:
    if col in bookings.columns:
        bookings[col] = pd.to_numeric(bookings[col], errors="coerce")

hotels["hotel_name"] = (
    hotels["brand"].astype(str).str.strip() + " - " +
    hotels["city"].astype(str).str.strip() + " (" +
    hotels["hotel_id"].astype(str).str.strip() + ")"
)

# =========================
# 3. VALIDATE COLUMNS
# =========================
required_booking_cols = {
    "user_id", "hotel_id", "trip_purpose", "party_type", "price_paid", "rating"
}
required_hotel_cols = {
    "hotel_id", "brand", "tier", "address", "city",
    "price_range_low", "price_range_high", "amenities"
}

if not required_booking_cols.issubset(bookings.columns):
    raise ValueError(f"Bookings file must contain columns: {required_booking_cols}")

if not required_hotel_cols.issubset(hotels.columns):
    raise ValueError(f"Hotels file must contain columns: {required_hotel_cols}")

# =========================
# 4. HELPER FUNCTIONS
# =========================
def safe_lower(x):
    return str(x).strip().lower()

def min_max_to_percent(series):
    if len(series) == 0:
        return series
    s_min = series.min()
    s_max = series.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_min == s_max:
        return pd.Series([100.0] * len(series), index=series.index)
    return ((series - s_min) / (s_max - s_min) * 100).clip(0, 100)

def get_available_cities():
    return sorted(hotels["city"].dropna().astype(str).unique())

def get_available_tiers():
    return sorted(hotels["tier"].dropna().astype(str).unique())

# =========================
# 5. BASE HOTEL POPULARITY SCORE
# =========================
hotel_stats = (
    bookings.groupby("hotel_id", as_index=False)
    .agg(
        bookings_count=("hotel_id", "count"),
        avg_rating=("rating", "mean"),
        avg_price_paid=("price_paid", "mean")
    )
)

hotel_stats["bookings_count"] = hotel_stats["bookings_count"].fillna(0)
hotel_stats["avg_rating"] = hotel_stats["avg_rating"].fillna(0)
hotel_stats["avg_price_paid"] = hotel_stats["avg_price_paid"].fillna(0)

hotel_stats["base_score"] = (
    hotel_stats["bookings_count"] * 0.4 +
    hotel_stats["avg_rating"] * 20
)

hotels_ranked = hotels.merge(hotel_stats, on="hotel_id", how="left")
for col in ["bookings_count", "avg_rating", "avg_price_paid", "base_score"]:
    hotels_ranked[col] = hotels_ranked[col].fillna(0)

# =========================
# 6. USER INPUT FILTER FUNCTIONS
# =========================
def filter_by_city(df, preferred_city=None):
    if preferred_city is None or df.empty:
        return df
    preferred_city = safe_lower(preferred_city)
    return df[df["city"].astype(str).str.strip().str.lower() == preferred_city]

def filter_by_budget(df, budget=None):
    if budget is None or df.empty:
        return df
    return df[df["price_range_low"] <= budget]

def filter_by_tier(df, preferred_tier=None):
    if preferred_tier is None or df.empty:
        return df
    preferred_tier = safe_lower(preferred_tier)
    return df[df["tier"].astype(str).str.strip().str.lower() == preferred_tier]

# =========================
# 7. BUILD LOOKALIKE / CF SEGMENT
# =========================
booking_with_hotel = bookings.merge(
    hotels[["hotel_id", "city", "brand", "tier", "amenities"]],
    on="hotel_id",
    how="left"
)

booking_with_hotel["city_l"] = booking_with_hotel["city"].map(safe_lower)
booking_with_hotel["trip_l"] = booking_with_hotel["trip_purpose"].map(safe_lower)
booking_with_hotel["party_l"] = booking_with_hotel["party_type"].map(safe_lower)

def get_similar_booking_subset(
    preferred_city=None,
    trip_purpose=None,
    party_type=None
):
    city_l = safe_lower(preferred_city) if preferred_city else None
    trip_l = safe_lower(trip_purpose) if trip_purpose else None
    party_l = safe_lower(party_type) if party_type else None

    attempts = [
        {
            "name": "city + trip_purpose + party_type",
            "mask": (
                (booking_with_hotel["city_l"] == city_l) &
                (booking_with_hotel["trip_l"] == trip_l) &
                (booking_with_hotel["party_l"] == party_l)
            ) if city_l and trip_l and party_l else None
        },
        {
            "name": "city + trip_purpose",
            "mask": (
                (booking_with_hotel["city_l"] == city_l) &
                (booking_with_hotel["trip_l"] == trip_l)
            ) if city_l and trip_l else None
        },
        {
            "name": "city + party_type",
            "mask": (
                (booking_with_hotel["city_l"] == city_l) &
                (booking_with_hotel["party_l"] == party_l)
            ) if city_l and party_l else None
        },
        {
            "name": "city only",
            "mask": (booking_with_hotel["city_l"] == city_l) if city_l else None
        },
        {
            "name": "trip_purpose + party_type",
            "mask": (
                (booking_with_hotel["trip_l"] == trip_l) &
                (booking_with_hotel["party_l"] == party_l)
            ) if trip_l and party_l else None
        },
        {
            "name": "trip_purpose only",
            "mask": (booking_with_hotel["trip_l"] == trip_l) if trip_l else None
        },
        {
            "name": "party_type only",
            "mask": (booking_with_hotel["party_l"] == party_l) if party_l else None
        },
        {
            "name": "global history",
            "mask": pd.Series([True] * len(booking_with_hotel), index=booking_with_hotel.index)
        }
    ]

    for attempt in attempts:
        if attempt["mask"] is None:
            continue
        subset = booking_with_hotel[attempt["mask"]].copy()
        if not subset.empty:
            return subset, attempt["name"]

    return booking_with_hotel.copy(), "global history"

def build_cf_scores(
    preferred_city=None,
    trip_purpose=None,
    party_type=None
):
    subset, match_level = get_similar_booking_subset(
        preferred_city=preferred_city,
        trip_purpose=trip_purpose,
        party_type=party_type
    )

    cf_scores = (
        subset.groupby("hotel_id", as_index=False)
        .agg(
            cf_booking_count=("hotel_id", "count"),
            cf_avg_rating=("rating", "mean"),
            cf_avg_price_paid=("price_paid", "mean"),
            cf_avg_stay=("length_of_stay", "mean")
        )
    )

    cf_scores["cf_booking_count"] = cf_scores["cf_booking_count"].fillna(0)
    cf_scores["cf_avg_rating"] = cf_scores["cf_avg_rating"].fillna(0)
    cf_scores["cf_avg_price_paid"] = cf_scores["cf_avg_price_paid"].fillna(0)
    cf_scores["cf_avg_stay"] = cf_scores["cf_avg_stay"].fillna(0)

    cf_scores["cf_score_raw"] = (
        cf_scores["cf_booking_count"] * 0.5 +
        cf_scores["cf_avg_rating"] * 20
    )

    return cf_scores, match_level, subset

# =========================
# 8. PERSONALIZATION / CONTENT SCORE
# =========================
def add_personalization_scores(
    df,
    budget=None,
    preferred_city=None,
    preferred_tier=None,
    trip_purpose=None,
    party_type=None,
    segment_subset=None
):
    if df.empty:
        return df

    df = df.copy()

    preferred_city_l = safe_lower(preferred_city) if preferred_city else None
    preferred_tier_l = safe_lower(preferred_tier) if preferred_tier else None

    df["city_match"] = 0
    df["tier_match"] = 0

    if preferred_city_l:
        df["city_match"] = (df["city"].map(safe_lower) == preferred_city_l).astype(int)

    if preferred_tier_l:
        df["tier_match"] = (df["tier"].map(safe_lower) == preferred_tier_l).astype(int)

    if budget is not None:
        midpoint = (df["price_range_low"] + df["price_range_high"]) / 2
        budget_diff = (midpoint - budget).abs()
        max_diff = budget_diff.max()

        if pd.notna(max_diff) and max_diff > 0:
            df["budget_fit_percent"] = (1 - (budget_diff / max_diff)) * 100
        else:
            df["budget_fit_percent"] = 100.0

        df.loc[df["price_range_low"] > budget, "budget_fit_percent"] *= 0.5
    else:
        df["budget_fit_percent"] = 100.0

    df["budget_fit_percent"] = df["budget_fit_percent"].clip(0, 100)

    if segment_subset is not None and not segment_subset.empty:
        segment_avg_paid = segment_subset["price_paid"].dropna().mean()
    else:
        segment_avg_paid = np.nan

    if pd.notna(segment_avg_paid):
        midpoint = (df["price_range_low"] + df["price_range_high"]) / 2
        price_diff = (midpoint - segment_avg_paid).abs()
        max_diff = price_diff.max()

        if pd.notna(max_diff) and max_diff > 0:
            df["segment_price_fit_percent"] = ((1 - (price_diff / max_diff)) * 100).clip(0, 100)
        else:
            df["segment_price_fit_percent"] = 100.0
    else:
        df["segment_price_fit_percent"] = 100.0

    df["content_score_raw"] = (
        df["city_match"] * 55 +
        df["tier_match"] * 20 +
        (df["budget_fit_percent"] / 100) * 15 +
        (df["segment_price_fit_percent"] / 100) * 10
    )

    return df

# =========================
# 9. RECOMMENDER
# =========================
def recommend_hotels(
    budget=None,
    preferred_city=None,
    preferred_tier=None,
    top_n=5
):
    recs = hotels_ranked.copy()

    cf_scores, match_level, segment_subset = build_cf_scores(
        preferred_city=preferred_city,
        trip_purpose=None,
        party_type=None
    )

    recs = recs.merge(cf_scores, on="hotel_id", how="left")
    for col in ["cf_booking_count", "cf_avg_rating", "cf_avg_price_paid", "cf_avg_stay", "cf_score_raw"]:
        recs[col] = recs[col].fillna(0)

    recs = filter_by_city(recs, preferred_city)
    recs = filter_by_budget(recs, budget)
    recs = filter_by_tier(recs, preferred_tier)

    if recs.empty:
        fallback_attempts = [
            {"tier": None, "budget": budget, "city": preferred_city},
            {"tier": None, "budget": None, "city": preferred_city},
            {"tier": None, "budget": None, "city": None},
        ]

        for attempt in fallback_attempts:
            tmp = hotels_ranked.copy()
            tmp = tmp.merge(cf_scores, on="hotel_id", how="left")
            for col in ["cf_booking_count", "cf_avg_rating", "cf_avg_price_paid", "cf_avg_stay", "cf_score_raw"]:
                tmp[col] = tmp[col].fillna(0)

            tmp = filter_by_city(tmp, attempt["city"])
            tmp = filter_by_budget(tmp, attempt["budget"])
            tmp = filter_by_tier(tmp, attempt["tier"])

            if not tmp.empty:
                recs = tmp.copy()
                break

    if recs.empty:
        return pd.DataFrame()

    recs = add_personalization_scores(
        recs,
        budget=budget,
        preferred_city=preferred_city,
        preferred_tier=preferred_tier,
        trip_purpose=None,
        party_type=None,
        segment_subset=segment_subset
    )

    recs["cf_score_percent"] = min_max_to_percent(recs["cf_score_raw"]).round(0)
    recs["content_score_percent"] = min_max_to_percent(recs["content_score_raw"]).round(0)
    recs["base_score_percent"] = min_max_to_percent(recs["base_score"]).round(0)

    recs["match_percent"] = (
        0.50 * recs["cf_score_percent"] +
        0.30 * recs["content_score_percent"] +
        0.20 * recs["base_score_percent"]
    ).round(0)

    def explain(row):
        reasons = []

        if row["city_match"]:
            reasons.append("matches city")
        if row["tier_match"]:
            reasons.append("matches tier")
        if row["budget_fit_percent"] >= 85:
            reasons.append("fits budget")
        elif row["budget_fit_percent"] >= 60:
            reasons.append("close to budget")
        if row["cf_booking_count"] > 0:
            reasons.append("popular with similar travelers")
        if row["cf_avg_rating"] >= 4:
            reasons.append("strong historical rating")

        if not reasons:
            reasons.append("good overall match")

        return ", ".join(reasons)

    recs["why_recommended"] = recs.apply(explain, axis=1)
    recs["cf_match_level"] = match_level

    recs = recs.sort_values(
        by=[
            "match_percent",
            "cf_score_percent",
            "content_score_percent",
            "base_score_percent"
        ],
        ascending=[False, False, False, False]
    )

    return recs.head(top_n)

# =========================
# 10. APP-READY OUTPUT FUNCTION
# =========================
def generate_hotel_recommendations(
    budget=None,
    preferred_city=None,
    preferred_tier=None,
    hotel_top_n=5
):
    if preferred_city is not None:
        city_exists = hotels["city"].astype(str).str.strip().str.lower().eq(
            preferred_city.strip().lower()
        ).any()

        if not city_exists:
            return {
                "success": False,
                "message": f"No hotels found in '{preferred_city}' in this dataset.",
                "hotel_recs": pd.DataFrame(),
                "display_df": pd.DataFrame()
            }

    hotel_recs = recommend_hotels(
        budget=budget,
        preferred_city=preferred_city,
        preferred_tier=preferred_tier,
        top_n=hotel_top_n
    )

    if hotel_recs.empty:
        return {
            "success": False,
            "message": "No hotels could be recommended from this dataset.",
            "hotel_recs": hotel_recs,
            "display_df": pd.DataFrame()
        }

    display_df = hotel_recs[
        [
            "hotel_name",
            "city",
            "brand",
            "tier",
            "address",
            "price_range_low",
            "price_range_high",
            "match_percent",
            "cf_match_level",
            "why_recommended"
        ]
    ].copy()

    display_df.columns = [
        "Hotel",
        "City",
        "Brand",
        "Tier",
        "Address",
        "Min Price",
        "Max Price",
        "Match %",
        "CF Match Level",
        "Why Recommended"
    ]

    display_df = display_df.reset_index(drop=True)

    return {
        "success": True,
        "message": f"We believe these {len(hotel_recs)} hotel{'s' if len(hotel_recs) != 1 else ''} would be a good match for you.",
        "hotel_recs": hotel_recs.reset_index(drop=True),
        "display_df": display_df
    }

# =========================
# 11. OPTIONAL NOTEBOOK DISPLAY HELPER
# =========================
def display_hotel_recommendations_console(display_df, message=None):
    if message:
        print("\n" + message)

    if display_df is None or display_df.empty:
        return

    print("\n=== HOTEL RECOMMENDATIONS ===")

    for i, row in display_df.iterrows():
        print(f"\n🔹 Option {i+1}: {row['Hotel']}")
        print(f"   📍 Location: {row['City']}")
        print(f"   🏨 Tier: {row['Tier']} | Brand: {row['Brand']}")
        print(f"   💰 Price: ${row['Min Price']} - ${row['Max Price']}")
        print(f"   ⭐ Match Score: {row['Match %']}%")
        print(f"   🤖 Why: {row['Why Recommended']}")
        print("-" * 60)
