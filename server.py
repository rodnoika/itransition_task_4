from contextlib import asynccontextmanager
from pathlib import Path
from collections import defaultdict, deque
from functools import lru_cache
from threading import BoundedSemaphore, Lock
from time import monotonic
import json
import os

import pandas as pd
import uvicorn

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from dotenv import load_dotenv

from analytics import Repository
from assistant import Assistant, GeminiPlanner, ProviderError
from schemas import AskRequest, PreviewRequest, ViewState


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env', override=False)
STORE_IDS = [1, 2, 3, 4, 5]
planner_lock = Lock()


def load_store(store_id):
    folder = Path(os.getenv('BOOKSTORE_DATA_DIR') or str(BASE_DIR)) / f"STORE{store_id}_cleaned"

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
    ) if not daily_revenue.empty else pd.DatetimeIndex([])

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
        "period_start": dates.min().strftime("%Y-%m-%d") if len(dates) else '—',
        "period_end": dates.max().strftime("%Y-%m-%d") if len(dates) else '—',
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


def load_repository():
    return Repository.load(Path(os.getenv('BOOKSTORE_DATA_DIR') or str(BASE_DIR)), BASE_DIR / 'title_people.json')


@lru_cache(maxsize=1)
def _cached_planner():
    return GeminiPlanner.from_settings(os.getenv('GEMINI_API_KEY', ''), os.getenv('GEMINI_MODEL') or 'gemini-3.1-pro-preview')


def load_planner():
    with planner_lock:
        return _cached_planner()


@lru_cache(maxsize=1)
def repository():
    return load_repository()


@asynccontextmanager
async def lifespan(app):
    repository.cache_clear()
    try:
        yield
    finally:
        repository.cache_clear()
        with planner_lock:
            try:
                if _cached_planner.cache_info().currsize:
                    _cached_planner().client.close()
            finally:
                _cached_planner.cache_clear()


app = FastAPI(
    title="Bookstore BI Dashboard",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

templates = Jinja2Templates(directory=BASE_DIR)
app.mount('/assets', StaticFiles(directory=BASE_DIR / 'public'), name='assets')
assistant_slots = BoundedSemaphore(2)
rate_lock = Lock()
recent_requests = defaultdict(deque)


def allowed_stores():
    values = {int(x.strip()) for x in os.getenv('ALLOWED_STORE_IDS', '1,2,3,4,5').split(',')}
    if not values or not values.issubset(set(STORE_IDS)):
        raise ValueError('ALLOWED_STORE_IDS must contain store numbers 1 to 5')
    return values


@app.middleware('http')
async def check_requests(request: Request, call_next):
    if request.url.path == '/healthz':
        return await call_next(request)
    if request.method == 'POST':
        expected = str(request.base_url).rstrip('/')
        if request.headers.get('origin') != expected or request.headers.get('content-type', '').split(';')[0] != 'application/json':
            return JSONResponse({'detail': 'Same-origin JSON request required'}, status_code=403)
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'same-origin'
    response.headers['X-Frame-Options'] = 'DENY'
    return response


@app.exception_handler(PermissionError)
async def forbidden(request, error):
    return JSONResponse({'detail': 'Store access denied'}, status_code=403)


@app.exception_handler(ValueError)
async def invalid_operation(request, error):
    detail = 'Gemini returned invalid function arguments; no filters were changed' if isinstance(error, ValidationError) else str(error)
    return JSONResponse({'detail': detail}, status_code=422)


@app.exception_handler(ProviderError)
async def provider_failed(request, error):
    return JSONResponse({'detail': str(error)}, status_code=error.status_code)


@app.get('/healthz')
def health():
    return {'status': 'ok'}


def get_dashboard(request, store_id):
    if store_id not in STORE_IDS:
        raise HTTPException(
            status_code=404,
            detail="Store not found"
        )

    data = repository()
    data.check_stores([store_id], allowed_stores())
    users, books, orders = data.select(store_id, ViewState().filters)
    return build_dashboard(store_id, users, books, orders)


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    store_id = min(allowed_stores())
    dashboard = get_dashboard(request, store_id)

    return templates.TemplateResponse(
        request=request,
        name="dashboard.htm",
        context={
            "dashboard": dashboard,
            "stores": sorted(allowed_stores()),
            "initial_state": ViewState(store_id=store_id).model_dump(mode='json'),
            "fragment": False,
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
            "stores": sorted(allowed_stores()),
            "initial_state": ViewState(store_id=store_id).model_dump(mode='json'),
            "fragment": False,
        }
    )


@app.get("/api/store/{store_id}")
def store_api(request: Request, store_id: int):
    return get_dashboard(request, store_id)


@app.get("/api/stores")
def stores_api(request: Request):
    result = []

    for store_id in sorted(allowed_stores()):
        dashboard = get_dashboard(request, store_id)

        result.append({
            "store_id": store_id,
            "store": dashboard["store"],
            "total_revenue_usd": dashboard["total_revenue_usd"],
            "unique_users": dashboard["unique_users"],
            "unique_author_sets": dashboard["unique_author_sets"]
        })

    return result


def view_payload(request, state, page=1, page_size=10):
    data = repository()
    preview = data.preview(state, allowed_stores(), page, page_size)
    users, books, orders = data.select(state.store_id, state.filters)
    dashboard = build_dashboard(state.store_id, users, books, orders)
    html = templates.get_template('dashboard.htm').render(request=request, dashboard=dashboard, stores=sorted(allowed_stores()), initial_state=state.model_dump(mode='json'), fragment=True)
    return {'preview': preview, 'dashboard': dashboard, 'dashboard_html': html}


@app.post('/api/view')
def filtered_view(request: Request, body: PreviewRequest):
    return view_payload(request, body.state, body.page, body.page_size)


@app.post('/api/assistant')
def ask_assistant(request: Request, body: AskRequest):
    data = repository()
    data.check_stores([body.state.store_id], allowed_stores())
    with rate_lock:
        history = recent_requests['analyst']
        now = monotonic()
        while history and history[0] <= now - 60:
            history.popleft()
        if len(history) >= 12:
            raise HTTPException(429, 'Too many AI requests. Try again in a minute.', headers={'Retry-After': '60'})
        history.append(now)
    if not assistant_slots.acquire(blocking=False):
        raise HTTPException(429, 'Assistant is busy. Try again shortly.')
    planner = None
    try:
        planner = load_planner()
        result = Assistant(data, planner).ask(body, allowed_stores())
        if result['action']:
            state = ViewState.model_validate(result['action']['state'])
            result['view'] = view_payload(request, state, page_size=body.page_size)
        return result
    finally:
        assistant_slots.release()


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
