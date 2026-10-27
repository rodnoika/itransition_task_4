import json

import pandas as pd
import pytest
from pydantic import ValidationError


@pytest.fixture
def repository():
    from analytics import Repository, StoreData

    users = pd.DataFrame({
        'user_id': [1, 2, 3, 4], 'canonical_user_id': [1, 1, 3, 4],
        'email': ['Sam@example.com', 'alias@example.com', 'pete@example.com', None],
    })
    books = pd.DataFrame({
        'book_id': [10, 11, 12],
        'title': ['Harry Potter', 'Plain book', 'Alice in Wonderland'],
        'authors': [['A', 'B'], ['A'], ['C']],
        'author_set': [json.dumps(['A', 'B']), json.dumps(['A']), json.dumps(['C'])],
        'genre': ['Fantasy', 'Science', 'Fantasy'],
        'has_person_name': [True, False, True],
        'person_names': [['Harry Potter'], [], ['Alice']],
    })
    orders = pd.DataFrame({
        'order_id': [1, 2, 3], 'book_id': [10, 11, 12],
        'user_id': [2, 3, 4], 'canonical_user_id': [1, 3, 4],
        'date': pd.to_datetime(['2026-01-01', '2026-01-03', '2026-01-03']),
        'quantity': [2, 1, 3], 'paid_price_cents': [1001, 999, 600],
    })
    return Repository({i: StoreData(users.copy(), books.copy(), orders.copy(), f'fixture-{i}') for i in [1, 2]})


def query(repository, **kwargs):
    from schemas import Query, ViewState
    return repository.query(Query(**kwargs), ViewState(), {1, 2})


def test_money_is_line_total_and_average_is_exact(repository):
    result = query(repository, metric='revenue')
    assert result['rows'] == [{'group': 'total', 'value': '26.00'}]
    assert result['evidence']['numerator_cents'] == 2600
    average = query(repository, metric='average_order_value')
    assert average['rows'][0]['value'] == '8.67'
    assert average['evidence']['denominator_orders'] == 3


def test_store_ids_do_not_merge_canonical_users(repository):
    assert query(repository, metric='users', store_ids=[1, 2])['rows'][0]['value'] == 6


def test_email_filter_selects_identity_and_all_its_alias_orders(repository):
    result = query(repository, metric='revenue', filters={'email_prefixes': ['s', 'P']})
    assert result['rows'][0]['value'] == '20.00'
    assert result['evidence']['matched_user_records'] == 3
    assert query(repository, metric='users', filters={'email_prefixes': ['s', 'p']})['rows'][0]['value'] == 2


def test_person_include_exclude_are_complements(repository):
    assert query(repository, metric='books', filters={'person_name': 'include'})['rows'][0]['value'] == 2
    assert query(repository, metric='books', filters={'person_name': 'exclude'})['rows'][0]['value'] == 1


def test_filter_is_literal_not_regex_or_executable(repository):
    assert query(repository, metric='books', filters={'title_contains': '.*'})['rows'][0]['value'] == 0
    with pytest.raises(ValidationError):
        query(repository, metric='revenue', sql='DROP TABLE users')


def test_date_bounds_empty_and_daily_group(repository):
    assert query(repository, metric='orders', filters={'start_date': '2026-01-03', 'end_date': '2026-01-03'})['rows'][0]['value'] == 2
    assert query(repository, metric='average_order_value', filters={'email_prefixes': ['z']})['rows'][0]['value'] is None
    assert query(repository, metric='revenue', group_by='date', sort='group')['rows'] == [
        {'group': '2026-01-01', 'value': '10.01'}, {'group': '2026-01-03', 'value': '15.99'},
    ]


def test_author_sales_credit_every_coauthor(repository):
    result = query(repository, metric='books_sold', group_by='author')
    assert result['rows'] == [{'group': 'A', 'value': 3}, {'group': 'C', 'value': 3}, {'group': 'B', 'value': 2}]


def test_forbidden_scope_and_bad_group_fail(repository):
    from schemas import Query, ViewState
    with pytest.raises(PermissionError):
        repository.query(Query(metric='users', store_ids=[2]), ViewState(), {1})
    with pytest.raises(ValueError):
        query(repository, metric='users', group_by='author')
    with pytest.raises(ValidationError):
        query(repository, metric='users', filters={'start_date': '2026-02-01', 'end_date': '2026-01-01'})


def test_preview_masks_pii_and_does_not_mutate_data(repository):
    from schemas import ViewState
    state = ViewState.model_validate({'filters': {'email_prefixes': ['s']}, 'tab': 'users'})
    result = repository.preview(state, {1})
    assert result['users']['total'] == 2
    assert result['users']['rows'][0]['email'] == 'S***@***'
    assert 'Sam@example.com' not in json.dumps(result)
    assert len(repository.stores[1].users) == 4


def test_missing_person_index_never_silently_returns_zero(repository):
    repository.stores[1].books = repository.stores[1].books.drop(columns=['has_person_name', 'person_names'])
    with pytest.raises(ValueError, match='PERSON'):
        query(repository, metric='books', filters={'person_name': 'include'})


def test_count_author_sets_uses_catalog_and_deduplicates_across_stores(repository):
    assert query(repository, metric='author_sets', store_ids=[1, 2])['rows'][0]['value'] == 3


def test_best_buyers_keeps_tied_alias_ids_and_exact_amount(repository):
    from schemas import Buyers, ViewState
    result = repository.best_buyers(Buyers(), ViewState(), {1})
    assert result['rows'] == [{'store_id': 1, 'user_ids': [1, 2], 'spent_usd': '10.01'}]
