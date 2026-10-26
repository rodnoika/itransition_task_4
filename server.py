from contextlib import asynccontextmanager
from pathlib import Path
import json

import pandas as pd
import uvicorn

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent
STORE_IDS = [1, 2, 3, 4, 5]


def load_store(store_id):
    folder = BASE_DIR / f"STORE{store_id}_cleaned"

    users_path = folder / "users_clean.parquet"
    books_path = folder / "books_clean.parquet"
    orders_path = folder / "orders_clean.parquet"

    for path in [users_path, books_path, orders_path]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    users = pd.read_parquet(users_path)
    books = pd.read_parquet(books_path)
    orders = pd.read_parquet(orders_path)

    return users, books, orders


def build_dashboard(store_id, users, books, orders):
    if orders.empty:
        raise ValueError(f"STORE{store_id} has no orders")

    orders = orders.copy()
    orders["date"] = pd.to_datetime(orders["date"]).dt.normalize()

    daily_revenue = (
        orders.groupby("date")["paid_price_cents"]
        .sum()
        .sort_index()
    )

    top_5 = (
        daily_revenue
        .rename("revenue_cents")
        .reset_index()
        .sort_values(
            ["revenue_cents", "date"],
            ascending=[False, True]
        )
        .head(5)
    )

    top_days = [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "revenue_usd": int(row.revenue_cents) / 100
        }
        for row in top_5.itertuples(index=False)
    ]

    unique_users = users["canonical_user_id"].nunique()
    unique_author_sets = books["author_set"].nunique()

    author_sales = orders[["book_id", "quantity"]].merge(
        books[["book_id", "author_set"]],
        on="book_id",
        how="left",
        validate="many_to_one"
    )

    if author_sales["author_set"].isna().any():
        raise ValueError(
            f"STORE{store_id} contains orders with unknown books"
        )

    sold_by_set = (
        author_sales.groupby("author_set")["quantity"]
        .sum()
    )

    best_set_sales = sold_by_set.max()

    popular_sets = [
        {
            "names": json.loads(author_set),
            "books_sold": int(quantity)
        }
        for author_set, quantity in sold_by_set[
            sold_by_set == best_set_sales
        ].items()
    ]

    individual_sales = (
        orders[["book_id", "quantity"]]
        .merge(
            books[["book_id", "authors"]],
            on="book_id",
            how="left",
            validate="many_to_one"
        )
        .explode("authors")
    )

    sold_by_author = (
        individual_sales.groupby("authors")["quantity"]
        .sum()
    )

    best_author_sales = sold_by_author.max()

    popular_individuals = [
        {
            "name": str(author),
            "books_sold": int(quantity)
        }
        for author, quantity in sold_by_author[
            sold_by_author == best_author_sales
        ].items()
    ]

    spending = (
        orders.groupby("canonical_user_id")["paid_price_cents"]
        .sum()
    )

    max_spending = spending.max()
    best_buyers = []

    for canonical_id in spending[spending == max_spending].index:
        ids = (
            users.loc[
                users["canonical_user_id"] == canonical_id,
                "user_id"
            ]
            .sort_values()
            .astype(int)
            .tolist()
        )

        best_buyers.append({
            "ids": ids,
            "spent_usd": int(max_spending) / 100
        })

    dates = pd.date_range(
        daily_revenue.index.min(),
        daily_revenue.index.max(),
        freq="D"
    )

    full_revenue = daily_revenue.reindex(
        dates,
        fill_value=0
    )

    chart = {
        "dates": [
            date.strftime("%Y-%m-%d")
            for date in dates
        ],
        "revenue_usd": [
            int(value) / 100
            for value in full_revenue
        ]
    }

    return {
        "store_id": store_id,
        "store": f"STORE{store_id}",
        "period_start": dates.min().strftime("%Y-%m-%d"),
        "period_end": dates.max().strftime("%Y-%m-%d"),
        "total_revenue_usd": int(
            orders["paid_price_cents"].sum()
        ) / 100,
        "order_count": len(orders),
        "books_sold": int(orders["quantity"].sum()),
        "unique_users": int(unique_users),
        "source_user_records": len(users),
        "unique_author_sets": int(unique_author_sets),
        "top_days": top_days,
        "popular_sets": popular_sets,
        "popular_individuals": popular_individuals,
        "best_buyers": best_buyers,
        "chart": chart
    }


@asynccontextmanager
async def lifespan(app):
    dashboards = {}

    for store_id in STORE_IDS:
        users, books, orders = load_store(store_id)

        dashboards[store_id] = build_dashboard(
            store_id,
            users,
            books,
            orders
        )

    app.state.dashboards = dashboards

    yield


app = FastAPI(
    title="Bookstore BI Dashboard",
    lifespan=lifespan
)

templates = Jinja2Templates(directory=BASE_DIR)


def get_dashboard(request, store_id):
    if store_id not in STORE_IDS:
        raise HTTPException(
            status_code=404,
            detail="Store not found"
        )

    return request.app.state.dashboards[store_id]


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    dashboard = get_dashboard(request, 1)

    return templates.TemplateResponse(
        request=request,
        name="dashboard.htm",
        context={
            "dashboard": dashboard,
            "stores": STORE_IDS
        }
    )


@app.get("/store/{store_id}", response_class=HTMLResponse)
def store_dashboard(request: Request, store_id: int):
    dashboard = get_dashboard(request, store_id)

    return templates.TemplateResponse(
        request=request,
        name="dashboard.htm",
        context={
            "dashboard": dashboard,
            "stores": STORE_IDS
        }
    )


@app.get("/api/store/{store_id}")
def store_api(request: Request, store_id: int):
    return get_dashboard(request, store_id)


@app.get("/api/stores")
def stores_api(request: Request):
    result = []

    for store_id in STORE_IDS:
        dashboard = get_dashboard(request, store_id)

        result.append({
            "store_id": store_id,
            "store": dashboard["store"],
            "total_revenue_usd": dashboard["total_revenue_usd"],
            "unique_users": dashboard["unique_users"],
            "unique_author_sets": dashboard["unique_author_sets"]
        })

    return result


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )