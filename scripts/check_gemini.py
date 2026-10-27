from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from assistant import GeminiPlanner
from schemas import Query, ViewState, DashboardAction


def main():
    load_dotenv(ROOT / '.env')
    planner = GeminiPlanner.from_settings(os.getenv('GEMINI_API_KEY', ''), os.getenv('GEMINI_MODEL') or 'gemini-3.1-pro-preview')
    try:
        info = planner.client.models.get(model=planner.model)
        print(f'Model is accessible: {info.name}')
        tests = [
            ('How many unique users in the current store?', 'query_data', Query),
            ('Select users with emails starting with s or p', 'set_dashboard_view', DashboardAction),
            ('Keep only books with people names in their titles', 'set_dashboard_view', DashboardAction),
        ]
        for message, expected, schema in tests:
            calls = planner.generate(message, ViewState(), {1, 2, 3, 4, 5})
            if len(calls) != 1 or calls[0][0] != expected:
                raise SystemExit(f'Unexpected tool for {message}: {calls}')
            arguments = schema.model_validate(calls[0][1])
            if message.startswith('How many') and arguments.metric != 'users':
                raise SystemExit('User count was mapped incorrectly')
            if message.startswith('Select') and (arguments.tab != 'users' or set(arguments.filters.email_prefixes) != {'s', 'p'}):
                raise SystemExit('Email command was mapped incorrectly')
            if message.startswith('Keep') and (arguments.tab != 'books' or arguments.filters.person_name != 'include'):
                raise SystemExit('PERSON command was mapped incorrectly')
            print(f'PASS: {message}')
        print('Gemini checks passed. Only synthetic questions and empty dashboard state were sent.')
    finally:
        planner.client.close()


if __name__ == '__main__':
    main()
