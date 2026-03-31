import json
import logging
import os
from typing import List

from service.memory.llm import llm_chat
from tenacity import retry, stop_after_attempt, wait_exponential

MEMORY_CATEGORIZATION_PROMPT = """Your task is to assign each piece of information (or "memory") to one or more of the following categories. Feel free to use multiple categories per item when appropriate.

- Personal: family, friends, home, hobbies, lifestyle
- Relationships: social network, significant others, colleagues
- Preferences: likes, dislikes, habits, favorite media
- Health: physical fitness, mental health, diet, sleep
- Travel: trips, commutes, favorite places, itineraries
- Work: job roles, companies, projects, promotions
- Education: courses, degrees, certifications, skills development
- Projects: to-dos, milestones, deadlines, status updates
- AI, ML & Technology: infrastructure, algorithms, tools, research
- Technical Support: bug reports, error logs, fixes
- Finance: income, expenses, investments, billing
- Shopping: purchases, wishlists, returns, deliveries
- Legal: contracts, policies, regulations, privacy
- Entertainment: movies, music, games, books, events
- Messages: emails, SMS, alerts, reminders
- Customer Support: tickets, inquiries, resolutions
- Product Feedback: ratings, bug reports, feature requests
- News: articles, headlines, trending topics
- Organization: meetings, appointments, calendars
- Goals: ambitions, KPIs, long-term objectives

Guidelines:
- Return only the categories under 'categories' key in the JSON format.
- If you cannot categorize the memory, return an empty list with key 'categories'.
- Don't limit yourself to the categories listed above only. Feel free to create new categories based on the memory. Make sure that it is a single phrase.
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    try:
        content = llm_chat(
            messages=[
                {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT + "\n\nRespond ONLY with valid JSON in this exact format: {\"categories\": [\"category1\", \"category2\"]}"},
                {"role": "user", "content": memory},
            ],
            json_mode=True,
            options={"temperature": 0},
            timeout=30,
        )

        # Strip code fences
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed_data = json.loads(content)

        if isinstance(parsed_data, dict) and "categories" in parsed_data:
            categories = parsed_data["categories"]
        elif isinstance(parsed_data, list):
            categories = parsed_data
        else:
            logging.warning(f"Unexpected response format: {parsed_data}")
            return []

        return [cat.strip().lower() for cat in categories if cat]

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON response: {e}")
        return []
    except Exception as e:
        logging.error(f"Failed to get categories: {e}")
        return []
