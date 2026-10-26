from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations
import json
import re

import pandas as pd
import yaml


store_id = 5
folder = Path("data") / f"STORE{store_id}"

users = pd.read_csv(
    folder / "users.csv",
    dtype="string",
    keep_default_na=False
)

with (folder / "books.yaml").open(encoding="utf-8") as file:
    books = pd.DataFrame(yaml.safe_load(file))

orders = pd.read_parquet(folder / "orders.parquet")

users = users.rename(columns={"id": "user_id"})
books = books.rename(columns=lambda col: col.lstrip(":"))
books = books.rename(columns={"id": "book_id"})
orders = orders.rename(columns={"id": "order_id"})

users = users.drop_duplicates().reset_index(drop=True)
books = books.drop_duplicates().reset_index(drop=True)
orders = orders.drop_duplicates().reset_index(drop=True)


def clean_text(column):
    result = (
        column.astype("string")
        .str.normalize("NFKC")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    missing = result.str.casefold().isin(["", "null"])
    return result.mask(missing, pd.NA)


for col in ["name", "address", "phone", "email"]:
    users[col] = clean_text(users[col])

for col in ["title", "author", "genre", "publisher", "year"]:
    books[col] = clean_text(books[col])

books["publisher"] = books["publisher"].replace("-", pd.NA)
books["year"] = books["year"].replace(["-", "-1", "0"], pd.NA)

orders["shipping"] = clean_text(orders["shipping"])

users["phone"] = users["phone"].str.replace(
    r"[\s().-]", "", regex=True
)
users["email"] = users["email"].str.casefold()

users["user_id"] = pd.to_numeric(
    users["user_id"], errors="raise"
).astype("Int64")

books["book_id"] = pd.to_numeric(
    books["book_id"], errors="raise"
).astype("Int64")

books["year"] = pd.to_numeric(
    books["year"], errors="raise"
).astype("Int64")

for col in ["order_id", "user_id", "book_id", "quantity"]:
    orders[col] = pd.to_numeric(
        orders[col], errors="raise"
    ).astype("Int64")

for table, id_col in [
    (users, "user_id"),
    (books, "book_id"),
    (orders, "order_id")
]:
    assert table[id_col].notna().all(), f"Missing {id_col}"
    assert table[id_col].is_unique, f"Duplicate {id_col}"


books["authors"] = books["author"].str.split(",").map(
    lambda names: [name.strip() for name in names]
)

books["author_set"] = books["authors"].map(
    lambda names: json.dumps(
        sorted(set(names)), ensure_ascii=False
    )
).astype("string")


orders["timestamp_raw"] = orders["timestamp"].astype("string")

timestamps = (
    clean_text(orders["timestamp_raw"])
    .str.replace(r"(?i)\b([ap])\.m\.", r"\1m", regex=True)
    .str.replace(r"[,;]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

dotted_dates = timestamps.str.match(
    r"^\d{1,2}\.\d{2}\.\d{4} "
).fillna(False)

parsed_dates = pd.Series(
    pd.NaT,
    index=orders.index,
    dtype="datetime64[ns]"
)

parsed_dates.loc[dotted_dates] = pd.to_datetime(
    timestamps.loc[dotted_dates],
    format="%d.%m.%Y %H:%M:%S",
    errors="raise"
)

parsed_dates.loc[~dotted_dates] = pd.to_datetime(
    timestamps.loc[~dotted_dates],
    format="mixed",
    dayfirst=False,
    errors="raise"
)

orders["timestamp"] = parsed_dates
orders["date"] = orders["timestamp"].dt.normalize()


def price_in_usd(value):
    text = re.sub(r"\s+", "", str(value)).upper()
    text = text.replace("USD", "$").replace("EUR", "€")

    text = re.sub(
        r"^([$€])(\d+)¢(\d{2})$",
        r"\1\2.\3",
        text
    )

    text = re.sub(
        r"^(\d+)([$€])(\d{2})¢$",
        r"\2\1.\3",
        text
    )

    match = re.fullmatch(
        r"([$€])(\d+(?:\.\d{0,2})?)|(\d+(?:\.\d{0,2})?)([$€])",
        text
    )

    if match is None:
        raise ValueError(f"Invalid price: {value!r}")

    currency = match.group(1) or match.group(4)
    amount = Decimal(match.group(2) or match.group(3))
    rate = Decimal("1.2") if currency == "€" else Decimal("1")

    return amount * rate


def to_cents(amount):
    return int(
        (amount * 100).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP
        )
    )


orders["unit_price_raw"] = orders["unit_price"].astype("string")

usd_prices = orders["unit_price_raw"].map(price_in_usd)

orders["unit_price_cents"] = usd_prices.map(
    to_cents
).astype("Int64")

orders["unit_price"] = (
    orders["unit_price_cents"].astype("Float64") / 100
)

orders["paid_price_cents"] = (
    orders["quantity"] * orders["unit_price_cents"]
)

orders["paid_price"] = (
    orders["paid_price_cents"].astype("Float64") / 100
)


fields = ["name", "address", "phone", "email"]
keys = users[["user_id", *fields]].copy()

for col in fields:
    keys[col] = keys[col].str.casefold()

parent = {
    int(user_id): int(user_id)
    for user_id in users["user_id"]
}


def find(user_id):
    user_id = int(user_id)

    while parent[user_id] != user_id:
        parent[user_id] = parent[parent[user_id]]
        user_id = parent[user_id]

    return user_id


for selected in combinations(fields, 3):
    selected = list(selected)
    candidates = keys.dropna(subset=selected)

    for _, ids in candidates.groupby(
        selected, sort=False
    )["user_id"]:
        first_id = int(ids.iloc[0])

        for user_id in ids.iloc[1:]:
            a = find(first_id)
            b = find(user_id)
            parent[max(a, b)] = min(a, b)


users["canonical_user_id"] = (
    users["user_id"].map(find).astype("Int64")
)

user_map = users.set_index("user_id")["canonical_user_id"]

orders["canonical_user_id"] = (
    orders["user_id"].map(user_map).astype("Int64")
)


assert users["name"].notna().all()
assert users["phone"].str.fullmatch(r"\d{10}", na=False).all()
assert users["email"].str.fullmatch(
    r"[^\s@]+@[^\s@]+\.[^\s@]+", na=False
).all()

assert books[["title", "author", "genre"]].notna().all().all()
assert books["authors"].map(lambda names: all(names)).all()

required = [
    "user_id", "book_id", "quantity", "timestamp", "paid_price"
]

assert orders[required].notna().all().all()
assert orders["quantity"].gt(0).all()
assert orders["unit_price_cents"].gt(0).all()
assert orders["user_id"].isin(users["user_id"]).all()
assert orders["book_id"].isin(books["book_id"]).all()
assert orders["canonical_user_id"].notna().all()


for name, table in [
    ("users", users),
    ("books", books),
    ("orders", orders)
]:
    print(name, table.shape)

    missing = table.isna().sum()
    print("Missing:", missing[missing > 0].to_dict())


output = Path(f"STORE{store_id}_cleaned")
output.mkdir(exist_ok=True)

users.to_parquet(output / "users_clean.parquet", index=False)
books.to_parquet(output / "books_clean.parquet", index=False)
orders.to_parquet(output / "orders_clean.parquet", index=False)

print(f"STORE{store_id} cleaned successfully")