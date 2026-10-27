from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
from pathlib import Path
import json

import pandas as pd

from schemas import Filters, Query, ViewState


def dollars(cents, denominator=1):
    if not denominator:
        return None
    return str((Decimal(int(cents)) / Decimal(100 * denominator)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))


def title_hash(books):
    rows = [(int(row.book_id), str(row.title)) for row in books.sort_values('book_id').itertuples()]
    return sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()


@dataclass
class StoreData:
    users: pd.DataFrame
    books: pd.DataFrame
    orders: pd.DataFrame
    version: str


class Repository:
    def __init__(self, stores):
        self.stores = stores

    @classmethod
    def load(cls, root: Path, index_path: Path):
        index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else {}
        stores = {}
        for store_id in range(1, 6):
            folder = root / f'STORE{store_id}_cleaned'
            paths = [folder / f'{table}_clean.parquet' for table in ['users', 'books', 'orders']]
            if any(not path.is_file() for path in paths):
                raise ValueError(f'Missing Parquet files in STORE{store_id}_cleaned')
            users, books, orders = [pd.read_parquet(path) for path in paths]
            required = [
                (users, ['user_id', 'canonical_user_id', 'email']),
                (books, ['book_id', 'title', 'authors', 'author_set', 'genre']),
                (orders, ['order_id', 'user_id', 'canonical_user_id', 'book_id', 'date', 'quantity', 'paid_price_cents']),
            ]
            for frame, columns in required:
                if not set(columns).issubset(frame.columns):
                    raise ValueError(f'STORE{store_id}: cleaned schema does not match')
                if frame[columns[0]].isna().any() or frame[columns[0]].duplicated().any():
                    raise ValueError(f'STORE{store_id}: duplicate or missing record IDs')
            for frame, columns in [(users, ['canonical_user_id']), (books, ['title', 'author_set']), (orders, ['canonical_user_id', 'book_id', 'date', 'quantity', 'paid_price_cents'])]:
                if frame[columns].isna().any().any():
                    raise ValueError(f'STORE{store_id}: missing required values')
            for column in ['quantity', 'paid_price_cents']:
                if not pd.api.types.is_integer_dtype(orders[column]):
                    raise ValueError(f'STORE{store_id}: {column} must contain integers')
            if not orders.book_id.isin(books.book_id).all() or not orders.user_id.isin(users.user_id).all():
                raise ValueError(f'STORE{store_id}: invalid order references')
            identities = users.set_index('user_id').canonical_user_id
            if not orders.user_id.map(identities).eq(orders.canonical_user_id).all():
                raise ValueError(f'STORE{store_id}: inconsistent canonical identities')
            orders = orders.copy()
            orders['date'] = pd.to_datetime(orders.date, errors='raise').dt.normalize()
            annotation = index.get('stores', {}).get(str(store_id), {})
            if annotation.get('title_hash') == title_hash(books):
                labels = annotation.get('people', {})
                if set(labels) == set(books.book_id.astype(str)) and all(isinstance(v, list) and all(isinstance(n, str) for n in v) for v in labels.values()):
                    books = books.copy()
                    books['person_names'] = books.book_id.map(lambda value: labels[str(value)])
                    books['has_person_name'] = books.person_names.map(bool)
            digest = sha256()
            for path in paths:
                digest.update(path.read_bytes())
            digest.update(json.dumps(annotation, sort_keys=True).encode())
            stores[store_id] = StoreData(users, books, orders, digest.hexdigest())
        return cls(stores)

    def check_stores(self, store_ids, allowed):
        if not set(store_ids).issubset(allowed):
            raise PermissionError('Store access denied')
        if not set(store_ids).issubset(self.stores):
            raise ValueError('Store not found')

    def select(self, store_id, filters):
        data = self.stores[store_id]
        users, books, orders = data.users, data.books, data.orders
        if filters.email_prefixes:
            matching = users.email.astype('string').str.strip().str.casefold().str.startswith(tuple(filters.email_prefixes), na=False)
            users = users[users.canonical_user_id.isin(users.loc[matching, 'canonical_user_id'])]
        if filters.title_contains:
            books = books[books.title.astype('string').str.casefold().str.contains(filters.title_contains.casefold(), regex=False, na=False)]
        if filters.author_contains:
            books = books[books.authors.map(lambda names: any(filters.author_contains.casefold() in str(name).casefold() for name in names))]
        if filters.genres:
            books = books[books.genre.astype('string').str.casefold().isin(filters.genres)]
        if filters.person_name != 'any':
            if 'has_person_name' not in books:
                raise ValueError('PERSON index missing or stale. Provide title_people.json matching the current catalogs.')
            books = books[books.has_person_name.eq(filters.person_name == 'include')]
        orders = orders[orders.book_id.isin(books.book_id) & orders.canonical_user_id.isin(users.canonical_user_id)]
        if filters.start_date:
            orders = orders[orders.date >= pd.Timestamp(filters.start_date)]
        if filters.end_date:
            orders = orders[orders.date <= pd.Timestamp(filters.end_date)]
        if filters.start_date or filters.end_date or filters.title_contains or filters.author_contains or filters.genres or filters.person_name != 'any':
            users = users[users.canonical_user_id.isin(orders.canonical_user_id)]
        if filters.start_date or filters.end_date or filters.email_prefixes:
            books = books[books.book_id.isin(orders.book_id)]
        return users, books, orders

    def evidence(self, store_ids, filters, selected):
        result = {
            'stores': store_ids, 'filters': filters.model_dump(mode='json'),
            'datasets': {str(i): self.stores[i].version for i in store_ids},
            'sources': ['users_clean.parquet', 'books_clean.parquet', 'orders_clean.parquet'],
            'matched_user_records': sum(len(u) for u, b, o in selected),
            'matched_books': sum(len(b) for u, b, o in selected),
            'matched_order_rows': sum(len(o) for u, b, o in selected),
            'identity_rule': 'canonical_user_id within each STORE; a matching email selects every alias of that identity',
            'money_rule': 'paid_price_cents is the line total; do not multiply by quantity',
            'person_rule': 'local PERSON annotations, not a guarantee that all names were recognized',
        }
        result['proof_id'] = sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:20]
        return result

    def query(self, args: Query, state: ViewState, allowed):
        store_ids = args.store_ids or [state.store_id]
        self.check_stores(store_ids, allowed)
        filters = args.filters if args.filters is not None else (state.filters if args.scope == 'current_view' else Filters())
        selected = [self.select(i, filters) for i in store_ids]
        frames = []
        for store_id, (users, books, orders) in zip(store_ids, selected):
            if args.metric in ['users', 'user_records']:
                if args.group_by not in ['none', 'store']:
                    raise ValueError('Users can only be grouped by store')
                frame = users.drop_duplicates('canonical_user_id') if args.metric == 'users' else users
            elif args.metric in ['books', 'author_sets']:
                if args.group_by in ['date', 'month']:
                    raise ValueError('Catalog books cannot be grouped by order date')
                frame = books
            else:
                frame = orders.merge(books[['book_id', 'title', 'genre', 'authors', 'author_set']], on='book_id', validate='many_to_one')
            frame = frame.copy()
            frame['store'] = f'STORE{store_id}'
            if args.group_by == 'none':
                frame['group'] = 'total'
            elif args.group_by == 'store':
                frame['group'] = frame.store
            elif args.group_by == 'date':
                frame['group'] = frame.date.dt.strftime('%Y-%m-%d')
            elif args.group_by == 'month':
                frame['group'] = frame.date.dt.strftime('%Y-%m')
            elif args.group_by == 'book':
                frame['group'] = frame.apply(lambda r: f'STORE{store_id} / {r.book_id} / {r.title}', axis=1) if not frame.empty else pd.Series(dtype='str')
            elif args.group_by == 'author':
                frame = frame.explode('authors')
                frame['group'] = frame.authors
            else:
                frame['group'] = frame[args.group_by]
            frames.append(frame)
        data = pd.concat(frames, ignore_index=True)
        groups = list(data.groupby('group', sort=True, dropna=False))
        if args.group_by == 'none' and not groups:
            groups = [('total', data)]
        raw_rows = []
        for group, frame in groups:
            if args.metric == 'revenue':
                value = sum(int(x) for x in frame.paid_price_cents)
            elif args.metric == 'average_order_value':
                value = Decimal(sum(int(x) for x in frame.paid_price_cents)) / Decimal(len(frame)) if len(frame) else None
            elif args.metric == 'books_sold':
                value = sum(int(x) for x in frame.quantity)
            elif args.metric == 'author_sets':
                value = int(frame.author_set.nunique())
            else:
                value = len(frame)
            numerator = sum(int(x) for x in frame.paid_price_cents) if args.metric == 'average_order_value' else 0
            raw_rows.append((str(group), value, len(frame), numerator))
        raw_rows.sort(key=lambda row: row[0])
        if args.sort != 'group':
            raw_rows.sort(key=lambda row: (row[1] is not None, row[1] if row[1] is not None else 0), reverse=args.sort == 'value_desc')
        rows = []
        for group, value, count, numerator in raw_rows[:args.limit]:
            if args.metric == 'revenue':
                value = dollars(value)
            elif args.metric == 'average_order_value':
                value = dollars(numerator, count)
            rows.append({'group': group, 'value': value})
        proof = self.evidence(store_ids, filters, selected)
        proof.update({'metric': args.metric, 'group_by': args.group_by, 'sort': args.sort,
                      'numerator_cents': sum(int(x) for u, b, o in selected for x in o.paid_price_cents),
                      'denominator_orders': sum(len(o) for u, b, o in selected),
                      'coauthor_rule': 'each coauthor receives full credit; author rows are not additive'})
        proof['query_id'] = sha256((proof['proof_id'] + args.model_dump_json()).encode()).hexdigest()[:20]
        return {'metric': args.metric, 'group_by': args.group_by, 'rows': rows, 'total_groups': len(raw_rows), 'truncated': len(raw_rows) > args.limit, 'evidence': proof}

    def best_buyers(self, args, state, allowed):
        store_ids = list(dict.fromkeys(args.store_ids)) or [state.store_id]
        self.check_stores(store_ids, allowed)
        filters = state.filters if args.scope == 'current_view' else Filters()
        selected = [self.select(i, filters) for i in store_ids]
        candidates = []
        for store_id, (users, books, orders) in zip(store_ids, selected):
            for canonical, frame in orders.groupby('canonical_user_id'):
                candidates.append((sum(int(x) for x in frame.paid_price_cents), store_id, canonical, users))
        maximum = max((row[0] for row in candidates), default=None)
        winners = [row for row in candidates if row[0] == maximum]
        rows = [{'store_id': i, 'user_ids': sorted(int(x) for x in u.loc[u.canonical_user_id == c, 'user_id']), 'spent_usd': dollars(v)} for v, i, c, u in winners[:50]]
        return {'rows': rows, 'total_winners': len(winners), 'truncated': len(winners) > 50, 'evidence': self.evidence(store_ids, filters, selected)}

    def preview(self, state: ViewState, allowed, page=1, page_size=10):
        self.check_stores([state.store_id], allowed)
        users, books, orders = self.select(state.store_id, state.filters)
        start = (page - 1) * page_size
        user_rows = []
        for row in users.sort_values('user_id').iloc[start:start + page_size].itertuples():
            email = str(row.email) if pd.notna(row.email) else ''
            user_rows.append({'user_id': int(row.user_id), 'canonical_user_id': int(row.canonical_user_id), 'email': email[:1] + '***@***' if email else '—'})
        book_rows = []
        for row in books.sort_values('book_id').iloc[start:start + page_size].itertuples():
            book_rows.append({'book_id': int(row.book_id), 'title': str(row.title), 'authors': [str(x) for x in row.authors], 'person_names': getattr(row, 'person_names', None)})
        return {'state': state.model_dump(mode='json'), 'page': page, 'page_size': page_size,
                'users': {'rows': user_rows, 'total': len(users), 'unique': int(users.canonical_user_id.nunique())},
                'books': {'rows': book_rows, 'total': len(books)},
                'summary': {'revenue_usd': dollars(sum(int(x) for x in orders.paid_price_cents)), 'orders': len(orders), 'books_sold': sum(int(x) for x in orders.quantity)},
                'evidence': self.evidence([state.store_id], state.filters, [(users, books, orders)])}
