import argparse
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analytics import Repository
from schemas import Query, ViewState


def main():
    parser = argparse.ArgumentParser(description='Validate local data and bundle inputs without deploying.')
    parser.add_argument('--data-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    data = Repository.load(args.data_dir, ROOT / 'title_people.json')
    for store_id, store in data.stores.items():
        if 'has_person_name' not in store.books:
            raise SystemExit(f'STORE{store_id}: PERSON index missing or stale; rebuild it first.')
        result = data.query(Query(metric='revenue'), ViewState(store_id=store_id), {store_id})
        print(f"STORE{store_id}: {len(store.users)} user records; {len(store.books)} books; {len(store.orders)} order rows; USD {result['rows'][0]['value']}")
    for name in ['server.py', 'dashboard.htm', 'schemas.py', 'analytics.py', 'assistant.py', 'requirements.txt', 'title_people.json', 'public/assistant.js']:
        if not (ROOT / name).is_file():
            raise SystemExit(f'Missing runtime file: {name}')
    config = json.loads((ROOT / 'vercel.json').read_text(encoding='utf-8'))
    assert config['functions']['server.py']['maxDuration'] >= 45
    print('Local data and runtime files passed. No deployment or external API calls were made.')
    if args.data_dir.resolve() != ROOT:
        print('For Vercel, copy all five STORE*_cleaned directories beside server.py into your private deployment source.')


if __name__ == '__main__':
    main()
