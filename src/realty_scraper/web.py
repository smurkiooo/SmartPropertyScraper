"""
Веб-интерфейс для RAG поиска квартир.

Запуск:
    uv run python -m realty_scraper.web
    Открой http://localhost:7860
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from realty_scraper.llm import LMStudioClient, _build_prompt, SYSTEM_PROMPT
from realty_scraper.search import ApartmentSearch

app = FastAPI()
_search = ApartmentSearch()
_llm = LMStudioClient()



HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Realty Search</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #f8f8f8;
      --surface: #ffffff;
      --border: #e0e0e0;
      --accent: #2563eb;
      --accent-hover: #1d4ed8;
      --text: #1a1a1a;
      --muted: #6b7280;
      --score-bg: #eff6ff;
      --score-color: #2563eb;
      --card-shadow: 0 1px 4px rgba(0,0,0,0.07);
      --radius: 12px;
    }

    body {
      background: var(--bg);
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    /* ── Hero ── */
    .hero {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      flex: 1;
      width: 100%;
      padding: 0 16px;
      transition: all 0.4s ease;
    }
    .hero.compact {
      flex: none;
      padding-top: 40px;
      padding-bottom: 24px;
    }

    .logo {
      font-size: 2.6rem;
      font-weight: 700;
      letter-spacing: -1px;
      color: var(--text);
      margin-bottom: 32px;
      transition: all 0.4s ease;
    }
    .logo span { color: var(--accent); }
    .hero.compact .logo { font-size: 1.6rem; margin-bottom: 20px; }

    /* ── Search bar ── */
    .search-wrap {
      width: 100%;
      max-width: 620px;
      position: relative;
    }
    .search-wrap input {
      width: 100%;
      padding: 16px 56px 16px 20px;
      border: 1.5px solid var(--border);
      border-radius: 999px;
      font-size: 1rem;
      background: var(--surface);
      color: var(--text);
      outline: none;
      box-shadow: var(--card-shadow);
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .search-wrap input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(37,99,235,0.12);
    }
    .search-wrap button {
      position: absolute;
      right: 8px;
      top: 50%;
      transform: translateY(-50%);
      background: var(--accent);
      border: none;
      border-radius: 999px;
      width: 38px;
      height: 38px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: background 0.2s;
    }
    .search-wrap button:hover { background: var(--accent-hover); }
    .search-wrap button svg { color: #fff; }

    /* ── Results ── */
    .results {
      width: 100%;
      max-width: 720px;
      padding: 0 16px 60px;
      display: none;
    }
    .results.visible { display: block; }

    /* ── Listing cards ── */
    .listings-title {
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 12px;
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 14px 18px;
      margin-bottom: 10px;
      display: flex;
      align-items: flex-start;
      gap: 14px;
      box-shadow: var(--card-shadow);
      text-decoration: none;
      color: inherit;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .card:hover {
      border-color: var(--accent);
      box-shadow: 0 2px 10px rgba(37,99,235,0.1);
    }
    .score-badge {
      background: var(--score-bg);
      color: var(--score-color);
      font-size: 0.72rem;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 999px;
      white-space: nowrap;
      margin-top: 2px;
    }
    .card-body { flex: 1; min-width: 0; }
    .card-title {
      font-size: 0.95rem;
      font-weight: 600;
      margin-bottom: 3px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .card-meta {
      font-size: 0.82rem;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .card-url {
      font-size: 0.75rem;
      color: var(--accent);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      margin-top: 4px;
    }

    /* ── LLM block ── */
    .llm-block {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px 22px;
      margin-top: 20px;
      box-shadow: var(--card-shadow);
      line-height: 1.7;
      font-size: 0.95rem;
    }
    .llm-block h3 {
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 14px;
    }
    .llm-text { white-space: pre-wrap; }
    .cursor {
      display: inline-block;
      width: 2px;
      height: 1em;
      background: var(--accent);
      margin-left: 2px;
      vertical-align: text-bottom;
      animation: blink 1s step-end infinite;
    }
    @keyframes blink { 50% { opacity: 0; } }

    /* ── Spinner ── */
    .spinner {
      width: 20px; height: 20px;
      border: 2px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      display: none;
      margin: 24px auto;
    }
    .spinner.active { display: block; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>

<div class="hero" id="hero">
  <div class="logo">Realty<span>Search</span></div>
  <div class="search-wrap">
    <input
      id="query"
      type="text"
      placeholder="Найди мне квартиру..."
      autocomplete="off"
    />
    <button onclick="doSearch()" title="Поиск">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2.5"
           stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
      </svg>
    </button>
  </div>
</div>

<div class="results" id="results">
  <div class="spinner" id="spinner"></div>
  <div id="cards"></div>
  <div class="llm-block" id="llm-block" style="display:none">
    <h3>Анализ от мини LLM: </h3>
    <div class="llm-text" id="llm-text"></div>
  </div>
</div>

<script>
  const queryInput = document.getElementById('query');

  queryInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') doSearch();
  });

  function fmt(price) {
    if (!price) return '—';
    return price.toLocaleString('ru-RU') + ' ₽';
  }

  async function doSearch() {
    const q = queryInput.value.trim();
    if (!q) return;

    const hero = document.getElementById('hero');
    const results = document.getElementById('results');
    const spinner = document.getElementById('spinner');
    const cards = document.getElementById('cards');
    const llmBlock = document.getElementById('llm-block');
    const llmText = document.getElementById('llm-text');

    hero.classList.add('compact');
    results.classList.add('visible');
    spinner.classList.add('active');
    cards.innerHTML = '';
    llmBlock.style.display = 'none';
    llmText.innerHTML = '';

    // ── 1. Получить карточки ──
    const resp = await fetch('/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q, top: 5 }),
    });
    const data = await resp.json();
    spinner.classList.remove('active');

    if (!data.results || data.results.length === 0) {
      cards.innerHTML = '<p style="color:var(--muted);padding:16px 0">Ничего не найдено.</p>';
      return;
    }

    // Заголовок
    const title = document.createElement('div');
    title.className = 'listings-title';
    title.textContent = `Найдено ${data.results.length} вариантов`;
    cards.appendChild(title);

    data.results.forEach(r => {
      const rooms = r.rooms ? `${r.rooms}-комн.` : 'Студия';
      const area  = r.area_total ? `${r.area_total} м²` : '';
      const floor = (r.floor && r.floors_total) ? `${r.floor}/${r.floors_total} эт.` : '';
      const metro = r.underground ? `м. ${r.underground}` : '';
      const meta  = [rooms, area, floor, metro].filter(Boolean).join(' · ');

      const a = document.createElement('a');
      a.className = 'card';
      a.href = r.url;
      a.target = '_blank';
      a.innerHTML = `
        <span class="score-badge">${(r.score * 100).toFixed(0)}%</span>
        <div class="card-body">
          <div class="card-title">${r.address || 'Москва'}</div>
          <div class="card-meta">${meta} &nbsp;·&nbsp; ${fmt(r.price)}</div>
          <div class="card-url">${r.url}</div>
        </div>`;
      cards.appendChild(a);
    });

    // ── 2. Стримить LLM ──
    llmBlock.style.display = 'block';
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    llmText.appendChild(cursor);

    const evtSource = new EventSource(`/stream?query=${encodeURIComponent(q)}&top=3`);
    evtSource.onmessage = e => {
      if (e.data === '[DONE]') {
        evtSource.close();
        cursor.remove();
        return;
      }
      const token = JSON.parse(e.data);
      cursor.insertAdjacentText('beforebegin', token);
      cursor.scrollIntoView({ block: 'nearest' });
    };
    evtSource.onerror = () => {
      evtSource.close();
      cursor.remove();
    };
  }
</script>
</body>
</html>"""


# ── API ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.post("/search")
async def search(body: dict):
    query: str = body.get("query", "")
    top: int = min(int(body.get("top", 3)), 3)
    results = _search.query(query, n_results=top)
    return {
        "results": [
            {
                "url": r.url,
                "score": round(r.score, 4),
                "price": r.price,
                "rooms": r.rooms,
                "area_total": r.area_total,
                "floor": r.floor,
                "floors_total": r.floors_total,
                "address": r.address,
                "underground": r.underground,
                "deal_type": r.deal_type,
            }
            for r in results
        ]
    }


@app.get("/stream")
async def stream(query: str, top: int = 3):
    top = min(top, 3)
    results = _search.query(query, n_results=top)

    async def generate() -> AsyncGenerator[str, None]:
        if not results:
            yield "data: [DONE]\n\n"
            return

        if not _llm.check_connection():
            msg = "llm недоступна... возможно выключен сервер"
            yield f"data: {json.dumps(msg)}\n\n"
            yield "data: [DONE]\n\n"
            return

        prompt = _build_prompt(query, results)
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        stream_resp = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=1024,
            stream=True,
        )
        for chunk in stream_resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps(delta)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def main():
    uvicorn.run("realty_scraper.web:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
