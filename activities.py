import streamlit as st
from google import genai

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# =========================
# AI AGENT: THINGS TO DO
# APP-READY VERSION
# =========================

def clean_activity_output(text):
    lines = text.strip().split("\n")
    cleaned_lines = []

    stop_phrases = [
        "would you like",
        "let me know",
        "i can also",
        "i can help",
        "do you want",
        "if you'd like"
    ]

    for line in lines:
        line_lower = line.strip().lower()
        if any(line_lower.startswith(phrase) for phrase in stop_phrases):
            break
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def get_things_to_do(hotel_name, address, city, trip_purpose=None, party_type=None):
    context_parts = []
    if trip_purpose:
        context_parts.append(f"trip purpose: {trip_purpose}")
    if party_type:
        context_parts.append(f"traveling as: {party_type}")
    context_str = (" The traveler's " + " and ".join(context_parts) + ".") if context_parts else ""

    prompt = (
        f"A traveler is staying at {hotel_name} located at {address}, {city}.{context_str} "
        f"Suggest exactly 5 things to do nearby.\n\n"
        f"For each activity include exactly these fields:\n"
        f"1. Name\n"
        f"2. Description\n"
        f"3. Distance and travel time from the hotel\n"
        f"4. Estimated cost or price range\n"
        f"5. Best time of day to visit\n"
        f"6. Why it suits this traveler\n\n"
        f"Rules:\n"
        f"- Prioritize activities within 30 minutes of the hotel\n"
        f"- Be specific to the location\n"
        f"- Keep each activity concise and practical\n"
        f"- Do NOT add bonus tips\n"
        f"- Do NOT add restaurant suggestions unless directly relevant\n"
        f"- Do NOT ask follow-up questions\n"
        f"- Do NOT add any closing sentence\n"
        f"- Return only the 5 activities in this exact format:\n\n"
        f"1. Activity Name\n"
        f"   Description: ...\n"
        f"   Distance/Time: ...\n"
        f"   Cost: ...\n"
        f"   Best Time: ...\n"
        f"   Why it fits: ...\n\n"
        f"2. Activity Name\n"
        f"   Description: ...\n"
        f"   Distance/Time: ...\n"
        f"   Cost: ...\n"
        f"   Best Time: ...\n"
        f"   Why it fits: ...\n\n"
        f"Continue the same structure through number 5 only."
    )

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
    )

    raw_text = response.text if response.text else ""
    return clean_activity_output(raw_text)
