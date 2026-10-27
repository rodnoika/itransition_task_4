from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


StoreId = Literal[1, 2, 3, 4, 5]
PageSize = Literal[10, 20, 30, 40, 50]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', str_max_length=2000)


class Filters(StrictModel):
    email_prefixes: list[str] = Field(default_factory=list, max_length=8)
    person_name: Literal['any', 'include', 'exclude'] = 'any'
    title_contains: str = Field(default='', max_length=120)
    author_contains: str = Field(default='', max_length=120)
    genres: list[str] = Field(default_factory=list, max_length=8)
    start_date: date | None = None
    end_date: date | None = None

    @field_validator('email_prefixes', 'genres')
    @classmethod
    def clean_values(cls, values):
        values = list(dict.fromkeys(value.strip().casefold() for value in values))
        if any(not value or len(value) > 80 for value in values):
            raise ValueError('Filter values must contain 1 to 80 characters')
        return values

    @model_validator(mode='after')
    def valid_range(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError('start_date must not be after end_date')
        return self


class ViewState(StrictModel):
    store_id: StoreId = 1
    tab: Literal['revenue', 'users', 'authors', 'books'] = 'revenue'
    filters: Filters = Field(default_factory=Filters)


class DashboardAction(ViewState):
    store_id: StoreId = Field(...)
    tab: Literal['revenue', 'users', 'authors', 'books'] = Field(...)
    filters: Filters = Field(...)


class Query(StrictModel):
    metric: Literal['users', 'user_records', 'books', 'author_sets', 'orders', 'revenue', 'books_sold', 'average_order_value']
    store_ids: list[StoreId] = Field(default_factory=list, max_length=5)
    scope: Literal['current_view', 'all_records'] = 'current_view'
    filters: Filters | None = None
    group_by: Literal['none', 'store', 'date', 'month', 'book', 'genre', 'author', 'author_set'] = 'none'
    sort: Literal['value_desc', 'value_asc', 'group'] = 'value_desc'
    limit: int = Field(default=20, ge=1, le=50, strict=True)

    @field_validator('store_ids')
    @classmethod
    def unique_stores(cls, values):
        return list(dict.fromkeys(values))


class Buyers(StrictModel):
    store_ids: list[StoreId] = Field(default_factory=list, max_length=5)
    scope: Literal['current_view', 'all_records'] = 'current_view'


class Clarify(StrictModel):
    reason: Literal['ambiguous_filter', 'unsupported_question', 'private_data']


class AskRequest(StrictModel):
    message: str = Field(min_length=1, max_length=1500)
    state: ViewState = Field(default_factory=ViewState)
    page_size: PageSize = 10

    @field_validator('message')
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError('Enter a question')
        return value.strip()


class PreviewRequest(StrictModel):
    state: ViewState
    page: int = Field(default=1, ge=1, le=100000, strict=True)
    page_size: PageSize = 10
