from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sqlite3
import json
import httpx
import random
import hashlib

app = FastAPI(title="Stock Topic Finder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "stock_topics.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS saved_topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            type TEXT,
            demand TEXT,
            competition TEXT,
            trend_percent REAL,
            opportunity_score INTEGER,
            keyword TEXT,
            saved_at TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            keyword TEXT NOT NULL,
            results_count INTEGER,
            searched_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()

init_db()

# --- Models ---
class SearchRequest(BaseModel):
    keyword: str
    topic_type: Optional[str] = "all"

class SaveTopicRequest(BaseModel):
    topic: str
    type: str
    demand: str
    competition: str
    trend_percent: float
    opportunity_score: int
    keyword: str

# --- Helpers ---
def get_user_id(api_key: str) -> str:
    """API key থেকে SHA256 hash বানাও — এটাই user_id, key কখনো store হবে না।"""
    return hashlib.sha256(api_key.strip().encode()).hexdigest()

def require_api_key(x_api_key: Optional[str]) -> tuple[str, str]:
    """Header থেকে key নাও, না থাকলে 401 দাও। (key, user_id) return করো।"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key missing. Please enter your Groq API key.")
    return x_api_key, get_user_id(x_api_key)

def compute_opportunity_score(demand, trend_pct, competition):
    demand_w = {"High": 40, "Medium": 25, "Low": 10}.get(demand, 20)
    trend_w = min(max(int(trend_pct / 2), -15), 30)
    comp_w = {"Low": 0, "Medium": -10, "High": -25}.get(competition, -10)
    return max(0, min(100, demand_w + trend_w + 40 + comp_w))

async def generate_topics_with_groq(keyword: str, api_key: str):
    prompt = f"""Generate exactly 20 Adobe Stock content topics for the keyword "{keyword}".
Mix of Main Topics (5) and Sub Topics (15).
For each topic return JSON with these fields:
- topic: specific descriptive topic name
- type: "Main Topic" or "Sub Topic"
- demand: "High", "Medium", or "Low"
- competition: "Low", "Medium", or "High"
- trend_percent: number between -15 and 45

Return ONLY a valid JSON array, no explanation, no markdown, no backticks."""

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2000,
            }
        )

    if resp.status_code == 401:
        raise HTTPException(status_code=401, detail="Invalid API key. Please check your Groq API key.")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Groq API error: {resp.status_code}")

    content = resp.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    topics = json.loads(content)
    result = []
    for i, t in enumerate(topics):
        trend_pct = float(t.get("trend_percent", random.uniform(-10, 30)))
        demand = t.get("demand", "Medium")
        competition = t.get("competition", "Medium")
        result.append({
            "id": i + 1,
            "topic": t["topic"],
            "type": t.get("type", "Sub Topic"),
            "demand": demand,
            "competition": competition,
            "trend_percent": round(trend_pct, 1),
            "opportunity_score": compute_opportunity_score(demand, trend_pct, competition),
        })
    return result

def generate_mock_topics(keyword):
    templates = [
        ("{kw} Modern Design", "Main Topic", "High", "Low", 28),
        ("{kw} Business Professional", "Sub Topic", "High", "Medium", 12),
        ("{kw} Flat Icon Set", "Sub Topic", "Medium", "Low", 18),
        ("{kw} Colorful Illustration", "Main Topic", "Medium", "Medium", 22),
        ("{kw} Minimal Logo", "Sub Topic", "High", "High", -5),
        ("{kw} Background Pattern", "Main Topic", "Medium", "Low", 15),
        ("{kw} Social Media Template", "Sub Topic", "Medium", "High", -8),
        ("{kw} Infographic Elements", "Sub Topic", "Medium", "Medium", 5),
        ("{kw} Eco Sustainable", "Sub Topic", "Medium", "Low", 30),
        ("{kw} Vintage Retro Style", "Sub Topic", "Low", "Low", 10),
    ]
    return [
        {
            "id": i + 1,
            "topic": tmpl.replace("{kw}", keyword.title()),
            "type": typ, "demand": dem, "competition": comp,
            "trend_percent": float(trend),
            "opportunity_score": compute_opportunity_score(dem, trend, comp),
        }
        for i, (tmpl, typ, dem, comp, trend) in enumerate(templates)
    ]

# --- Search ---
@app.post("/api/search")
async def search_topics(
    req: SearchRequest,
    x_api_key: Optional[str] = Header(default=None)
):
    api_key, user_id = require_api_key(x_api_key)
    used_fallback = False

    try:
        topics = await generate_topics_with_groq(req.keyword, api_key)
    except HTTPException:
        raise
    except Exception:
        topics = generate_mock_topics(req.keyword)
        used_fallback = True

    if req.topic_type and req.topic_type != "all":
        label = "Main Topic" if req.topic_type == "main" else "Sub Topic"
        topics = [t for t in topics if t["type"] == label]

    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO search_history (user_id, keyword, results_count) VALUES (?, ?, ?)",
        (user_id, req.keyword, len(topics))
    )
    conn.commit()
    conn.close()

    return {"keyword": req.keyword, "total": len(topics), "topics": topics, "used_fallback": used_fallback}

# --- Saved Topics ---
@app.get("/api/saved-topics")
def get_saved_topics(x_api_key: Optional[str] = Header(default=None)):
    _, user_id = require_api_key(x_api_key)
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM saved_topics WHERE user_id = ? ORDER BY saved_at DESC", (user_id,))
    rows = c.fetchall()
    conn.close()
    return {"topics": [dict(r) for r in rows]}

@app.post("/api/saved-topics")
def save_topic(req: SaveTopicRequest, x_api_key: Optional[str] = Header(default=None)):
    _, user_id = require_api_key(x_api_key)
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT id FROM saved_topics WHERE user_id = ? AND topic = ? AND keyword = ?",
        (user_id, req.topic, req.keyword)
    )
    if c.fetchone():
        conn.close()
        raise HTTPException(status_code=409, detail="Topic already saved")
    c.execute("""
        INSERT INTO saved_topics (user_id, topic, type, demand, competition, trend_percent, opportunity_score, keyword)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, req.topic, req.type, req.demand, req.competition, req.trend_percent, req.opportunity_score, req.keyword))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return {"success": True, "id": new_id}

@app.delete("/api/saved-topics/{topic_id}")
def delete_saved_topic(topic_id: int, x_api_key: Optional[str] = Header(default=None)):
    _, user_id = require_api_key(x_api_key)
    conn = get_db()
    c = conn.cursor()
    # শুধু নিজের topic delete করতে পারবে
    c.execute("DELETE FROM saved_topics WHERE id = ? AND user_id = ?", (topic_id, user_id))
    conn.commit()
    conn.close()
    return {"success": True}

# --- History ---
@app.get("/api/history")
def get_history(x_api_key: Optional[str] = Header(default=None)):
    _, user_id = require_api_key(x_api_key)
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM search_history WHERE user_id = ? ORDER BY searched_at DESC LIMIT 50",
        (user_id,)
    )
    rows = c.fetchall()
    conn.close()
    return {"history": [dict(r) for r in rows]}

@app.delete("/api/history")
def clear_history(x_api_key: Optional[str] = Header(default=None)):
    _, user_id = require_api_key(x_api_key)
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM search_history WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return {"success": True}

# --- Health ---
@app.get("/api/health")
def health():
    return {"status": "ok"}
