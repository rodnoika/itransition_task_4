import json

import pandas as pd
import pytest

from analytics import Repository, title_hash


def save_fixture(root):
    for store in range(1, 6):
        folder = root / f'STORE{store}_cleaned'
        folder.mkdir()
        pd.DataFrame({'user_id': [1], 'canonical_user_id': [1], 'email': ['a@example.com']}).to_parquet(folder / 'users_clean.parquet')
        pd.DataFrame({'book_id': [1], 'title': ['Alice'], 'authors': [['A']], 'author_set': ['["A"]'], 'genre': ['Fantasy']}).to_parquet(folder / 'books_clean.parquet')
        pd.DataFrame({'order_id': [1], 'book_id': [1], 'user_id': [1], 'canonical_user_id': [1], 'date': ['2026-01-01'], 'quantity': [1], 'paid_price_cents': [100]}).to_parquet(folder / 'orders_clean.parquet')


def test_stale_index_is_unavailable_not_silently_reused(tmp_path):
    save_fixture(tmp_path)
    index = tmp_path / 'title_people.json'
    index.write_text(json.dumps({'stores': {'1': {'title_hash': 'old', 'people': {'1': ['Alice']}}}}))
    repo = Repository.load(tmp_path, index)
    assert 'has_person_name' not in repo.stores[1].books


def test_matching_index_loads_complete_labels(tmp_path):
    save_fixture(tmp_path)
    books = pd.read_parquet(tmp_path / 'STORE1_cleaned/books_clean.parquet')
    index = tmp_path / 'title_people.json'
    index.write_text(json.dumps({'stores': {'1': {'title_hash': title_hash(books), 'people': {'1': ['Alice']}}}}))
    repo = Repository.load(tmp_path, index)
    assert repo.stores[1].books.has_person_name.tolist() == [True]


def test_inconsistent_identity_fails_instead_of_counting_wrong_people(tmp_path):
    save_fixture(tmp_path)
    path = tmp_path / 'STORE1_cleaned/orders_clean.parquet'
    orders = pd.read_parquet(path)
    orders['canonical_user_id'] = 2
    orders.to_parquet(path)
    with pytest.raises(ValueError, match='identities'):
        Repository.load(tmp_path, tmp_path / 'no-index.json')
