from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field


class User(BaseModel):
    name: str = Field(..., min_length=1)
    email: EmailStr
    risk_tolerance: str = Field(..., pattern=r"^(conservative|balanced|aggressive)$")
    goals: List[str] = []
    age: Optional[int] = None
    horizon_years: Optional[int] = None


class Holding(BaseModel):
    symbol: str = Field(..., min_length=1)
    quantity: float = Field(..., gt=0)
    avg_cost: float = Field(..., ge=0)
    sector: Optional[str] = None


class Portfolio(BaseModel):
    user_id: str
    holdings: List[Holding]


class ChatMessage(BaseModel):
    user_id: str
    message: str = Field(..., min_length=1)
    history: Optional[List[dict]] = None
