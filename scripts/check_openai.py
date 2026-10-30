from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from assistant import DEFAULT_OPENAI_MODEL, OpenAIPlanner, ProviderError
from schemas import Query, ViewState, DashboardAction


def main():
    load_dotenv(ROOT / '.env')
    planner = OpenAIPlanner.from_settings(os.getenv('OPENAI_API_KEY', ''), os.getenv('OPENAI_MODEL') or DEFAULT_OPENAI_MODEL)
    try:
        print(f'Checking OpenAI intent routing with {planner.model}')
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
        print('OpenAI checks passed. Only synthetic questions and empty dashboard state were sent.')
    finally:
        planner.client.close()


if __name__ == '__main__':
    try:
        main()
    except ProviderError as exc:
        raise SystemExit(str(exc)) from None
