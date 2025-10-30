from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import os
import httpx
from bson import ObjectId

from schemas import User, Portfolio, ChatMessage
from database import db, create_document, get_documents, update_document

app = FastAPI(title="AI Robo Advisory")

# CORS
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*" if FRONTEND_URL == "*" else FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/test")
async def test():
    try:
        db.list_collection_names()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/users/signin")
async def signin(user: User):
    existing = db["user"].find_one({"email": user.email})
    if existing:
        update_document(
            "user",
            {"_id": existing["_id"]},
            {
                "name": user.name,
                "risk_tolerance": user.risk_tolerance,
                "goals": user.goals,
                "age": user.age,
                "horizon_years": user.horizon_years,
            },
        )
        refreshed = db["user"].find_one({"email": user.email})
        refreshed["_id"] = str(refreshed["_id"])  # serialize
        return {"user": refreshed}
    created = create_document("user", user.model_dump())
    return {"user": created}


@app.post("/portfolio/save")
async def save_portfolio(p: Portfolio):
    # Upsert by user_id for simplicity
    existing = db["portfolio"].find_one({"user_id": p.user_id})
    if existing:
        db["portfolio"].update_one({"_id": existing["_id"]}, {"$set": p.model_dump()})
        updated = db["portfolio"].find_one({"_id": existing["_id"]})
        updated["_id"] = str(updated["_id"])  # serialize
        return {"portfolio": updated}
    created = create_document("portfolio", p.model_dump())
    return {"portfolio": created}


def _summarize_portfolio(portfolio: Dict[str, Any]) -> Dict[str, Any]:
    holdings = portfolio.get("holdings", [])
    total_cost = sum(h.get("avg_cost", 0) * h.get("quantity", 0) for h in holdings) or 1.0
    by_sector: Dict[str, float] = {}
    for h in holdings:
        sector = h.get("sector") or "Other"
        value = h.get("avg_cost", 0) * h.get("quantity", 0)
        by_sector[sector] = by_sector.get(sector, 0.0) + value
    sector_alloc = {k: round(v / total_cost * 100, 2) for k, v in by_sector.items()}
    top_positions = sorted(
        (
            {
                "symbol": h.get("symbol"),
                "weight": round((h.get("avg_cost", 0) * h.get("quantity", 0)) / total_cost * 100, 2),
            }
            for h in holdings
        ),
        key=lambda x: x["weight"],
        reverse=True,
    )[:5]
    return {
        "estimated_value": round(total_cost, 2),
        "sector_allocation": sector_alloc,
        "top_positions": top_positions,
        "holdings_count": len(holdings),
    }


def _heuristic_advice(user: Dict[str, Any], portfolio: Dict[str, Any]) -> str:
    risk = user.get("risk_tolerance", "balanced")
    summary = _summarize_portfolio(portfolio)
    value = summary["estimated_value"]
    sectors = summary["sector_allocation"]
    top_positions = summary["top_positions"]

    lines = [
        f"Here’s a quick, personalized checkup on your portfolio (~${value:,.0f}).",
        "Diversification:",
    ]
    if len(sectors) < 4:
        lines.append("- Consider adding exposure to more sectors to reduce concentration risk.")
    else:
        lines.append("- Sector mix looks reasonably diversified. Keep monitoring exposures.")

    if top_positions and top_positions[0]["weight"] > 25:
        lines.append("- Your largest position is above 25% of portfolio. Gradual trimming may reduce idiosyncratic risk.")

    lines.append("Risk alignment:")
    if risk == "conservative":
        lines.append("- Favor high-quality bonds, broad-market ETFs, and larger cash buffer (6–12 months expenses).")
    elif risk == "balanced":
        lines.append("- A mix of equities and bonds (e.g., 60/40) with periodic rebalancing can fit your profile.")
    else:
        lines.append("- Tilt toward equities/alternatives with disciplined DCA and volatility management.")

    lines.append("Next steps:")
    lines.append("- Set an automatic monthly investment and rebalance quarterly.")
    lines.append("- Map each goal to a time horizon and choose suitable vehicles (401k/IRA/taxable).")

    return "\n".join(lines)


async def _hf_complete(prompt: str) -> str:
    token = os.getenv("HF_API_TOKEN")
    if not token:
        return ""
    model = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 250, "temperature": 0.7}}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                full = data[0]["generated_text"]
                return full[len(prompt):].strip() if isinstance(full, str) and full.startswith(prompt) else full
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"]
            return ""
    except Exception:
        return ""


@app.post("/advice/analyze")
async def analyze(payload: Dict[str, Any]):
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    try:
        obj_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid user_id")
    user_doc = db["user"].find_one({"_id": obj_id})
    if not user_doc:
        raise HTTPException(status_code=404, detail="user not found")

    portfolio_doc = db["portfolio"].find_one({"user_id": user_id})
    if not portfolio_doc:
        raise HTTPException(status_code=404, detail="portfolio not found")

    summary = _summarize_portfolio(portfolio_doc)
    heuristic = _heuristic_advice(user_doc, portfolio_doc)

    sys_prompt = (
        "You are a helpful, compliant financial robo-advisor. "
        "Speak clearly and concisely. Avoid guaranteeing returns. \n\n"
        f"User risk tolerance: {user_doc.get('risk_tolerance')}\n"
        f"Goals: {', '.join(user_doc.get('goals', []))}\n"
        f"Portfolio summary: {summary}\n\n"
        "Provide a brief recommendation in 5-7 bullet points."
    )
    ai_text = await _hf_complete(sys_prompt)
    advice_text = ai_text or heuristic
    return {"summary": summary, "advice": advice_text}


@app.post("/chat")
async def chat(msg: ChatMessage):
    try:
        obj_id = ObjectId(msg.user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid user_id")
    user_doc = db["user"].find_one({"_id": obj_id})
    portfolio_doc = db["portfolio"].find_one({"user_id": msg.user_id}) or {"holdings": []}
    summary = _summarize_portfolio(portfolio_doc)

    system = (
        "You are an AI robo-advisor. Use simple language, include disclaimers, "
        "and tailor guidance to risk tolerance."
    )
    prompt = (
        f"{system}\n\nRisk: {user_doc.get('risk_tolerance')}\n"
        f"Goals: {', '.join(user_doc.get('goals', []))}\n"
        f"Portfolio: {summary}\n\n"
        f"User: {msg.message}\nAssistant:"
    )
    ai_text = await _hf_complete(prompt)
    if not ai_text:
        ai_text = (
            "Based on your profile, focus on staying diversified, rebalancing on a set schedule, "
            "and sizing positions according to your risk tolerance. Consider tax-advantaged accounts "
            "for long-term goals. I’m not a financial advisor; this is educational information."
        )
    return {"reply": ai_text}
