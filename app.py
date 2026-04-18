import streamlit as st
from recommender import (
    generate_hotel_recommendations,
    get_available_cities,
    get_available_tiers,
)
from activities import get_things_to_do

st.set_page_config(page_title="Accor Travel Assistant", layout="wide")

st.title("Accor Travel Assistant")
st.write("Find hotel recommendations and nearby things to do.")

# -------------------------
# Session state
# -------------------------
if "hotel_results" not in st.session_state:
    st.session_state.hotel_results = None

if "display_df" not in st.session_state:
    st.session_state.display_df = None

if "recommendation_message" not in st.session_state:
    st.session_state.recommendation_message = None

if "activities_result" not in st.session_state:
    st.session_state.activities_result = None

# -------------------------
# Step 1: Hotel recommendations
# -------------------------
st.header("1. Get hotel recommendations")

col1, col2, col3 = st.columns(3)

with col1:
    city_options = [""] + get_available_cities()
    selected_city = st.selectbox("Preferred city", city_options)

with col2:
    budget_input = st.text_input("Daily rate max budget", "")

with col3:
    tier_options = [""] + get_available_tiers()
    selected_tier = st.selectbox("Preferred tier", tier_options)

if st.button("Get hotel recommendations"):
    preferred_city = selected_city if selected_city != "" else None
    preferred_tier = selected_tier if selected_tier != "" else None

    try:
        budget = float(budget_input) if budget_input.strip() != "" else None
    except ValueError:
        st.error("Please enter a valid number for budget.")
        st.stop()

    result = generate_hotel_recommendations(
        budget=budget,
        preferred_city=preferred_city,
        preferred_tier=preferred_tier,
        hotel_top_n=5
    )

    if result["success"]:
        st.session_state.hotel_results = result["hotel_recs"]
        st.session_state.display_df = result["display_df"]
        st.session_state.recommendation_message = result["message"]
        st.session_state.activities_result = None
    else:
        st.session_state.hotel_results = None
        st.session_state.display_df = None
        st.session_state.recommendation_message = None
        st.session_state.activities_result = None
        st.error(result["message"])

# Show recommendations
if st.session_state.display_df is not None and not st.session_state.display_df.empty:
    st.success(st.session_state.recommendation_message)

    st.subheader("Recommended hotels")

    for i, row in st.session_state.display_df.iterrows():
        with st.container():
            st.markdown(f"### Option {i+1}: {row['Hotel']}")
            st.write(f"**Location:** {row['City']}")
            st.write(f"**Tier:** {row['Tier']} | **Brand:** {row['Brand']}")
            st.write(f"**Price:** ${row['Min Price']} - ${row['Max Price']}")
            st.write(f"**Match Score:** {row['Match %']}%")
            st.write(f"**Why Recommended:** {row['Why Recommended']}")
            st.divider()

# -------------------------
# Step 2: Things to do
# -------------------------
if st.session_state.hotel_results is not None and not st.session_state.hotel_results.empty:
    st.header("2. Get nearby activities")

    hotel_names = st.session_state.hotel_results["hotel_name"].tolist()
    selected_hotel_name = st.selectbox("Choose a recommended hotel", hotel_names)

    col4, col5 = st.columns(2)

    with col4:
        trip_purpose = st.selectbox(
            "What is the purpose of your trip?",
            ["", "Business", "Leisure", "Family"]
        )

    with col5:
        party_type = st.selectbox(
            "Who are you traveling with?",
            ["", "Solo", "Couple", "Family", "Group"]
        )

    if st.button("Get nearby activities"):
        selected_row = st.session_state.hotel_results[
            st.session_state.hotel_results["hotel_name"] == selected_hotel_name
        ].iloc[0]

        hotel_name = selected_row["hotel_name"]
        address = selected_row["address"]
        city = selected_row["city"]

        with st.spinner("Finding nearby activity suggestions..."):
            activities = get_things_to_do(
                hotel_name=hotel_name,
                address=address,
                city=city,
                trip_purpose=trip_purpose if trip_purpose != "" else None,
                party_type=party_type if party_type != "" else None
            )

        st.session_state.activities_result = {
            "hotel_name": hotel_name,
            "address": address,
            "city": city,
            "trip_purpose": trip_purpose if trip_purpose != "" else "Not provided",
            "party_type": party_type if party_type != "" else "Not provided",
            "activities": activities
        }

# Show activities
if st.session_state.activities_result is not None:
    data = st.session_state.activities_result

    st.subheader("Things to do near your hotel")
    st.write(f"**Selected Hotel:** {data['hotel_name']}")
    st.write(f"**Location:** {data['address']}, {data['city']}")
    st.write(f"**Trip Purpose:** {data['trip_purpose']}")
    st.write(f"**Traveling With:** {data['party_type']}")

    st.markdown("### 5 things to do nearby")
    st.text(data["activities"])

    st.success(f"We hope you enjoy your stay at {data['hotel_name']}!")
