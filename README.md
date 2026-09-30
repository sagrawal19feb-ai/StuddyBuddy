# 📘 StudyBuddy

**Your personal study companion — learn smarter, not harder.**

StudyBuddy is a smart, console-based educational chatbot written in Python 3.12+. It helps students plan study schedules, revise with quizzes and flashcards, get study tips and motivation, and answers questions it doesn't already know by extracting answers from the web. It is modular, object-oriented, fully documented, and works offline except for web search.

---

## ✨ Features

| Feature | What it does |
|---|---|
| 🧠 **Intent recognition** | Detects what you're asking with fuzzy matching (`rapidfuzz`), typo correction, contraction/slang handling (`"dont"` → `"do not"`, `"whts"` → `"what is"`), example-phrase matching and refusal detection (`"no quiz"`). |
| 💬 **Natural responses** | Hundreds of hand-written reply variants, chosen at random and never repeated twice in a row. Follow-up questions and conversation context. |
| 💡 **Study tips** | **200+ tips** in 10 categories (time management, memory, exams, focus, note-taking, revision, wellbeing, online learning, motivation, active learning). |
| 🌟 **Motivation** | **85 quotes** and encouragements. |
| 🗓 **Study scheduler** | Personalised weekly timetable from a few answers (hours, subjects, difficulty, exam date) — weighted by difficulty, weakness and exam proximity, with built-in Pomodoro rhythm. |
| 🌅 **Daily planner** | Balanced single-day routine: morning routine, study blocks, breaks, meals anchored to real clock times, exercise, revision, sleep. |
| 📝 **Quiz engine** | MCQ quizzes across **8 subjects × 3 difficulty levels**, shuffled questions and options, live scoring, **weak-topic analysis** and a performance report. Auto-generates extra questions from the knowledge base so quizzes never run out. |
| 🗃 **Flashcards** | Self-graded cards; missed cards return up to 3× then are flagged for practice (never loops forever). After a quiz it can turn **your missed questions** into a targeted flashcard session. |
| 🧮 **Maths solver** | `what is 12*8`, `15% of 200`, `square root of 144`, `2 to the power 10` — parsed safely with `ast`, so it can never execute code. |
| 🧠 **Study memory** | Teach it facts (`remember that mitochondria is the powerhouse of the cell`) and it recalls them later; listable and forgettable. |
| 🎯 **Study-method recommender** | `best way to study history` → a tailored technique per subject, data-driven from `subjects.json`. |
| 📖 **Answers from the web (primary)** | Every question is answered **live from the web** — the bot searches, fetches the top pages, extracts the relevant paragraphs and shows the **answer text inline** with its source (Wikipedia's clean summary first, page extraction as backup). No more just links. |
| 📚 **Offline fallback** | When the internet is unreachable, answers fall back to a local knowledge base (1000+ facts) with a clear "offline answer" note, and to curated links. |
| 📡 **Answers from the web** | Questions outside the knowledge base are answered by fetching search results, stripping boilerplate and extracting the best passage with its source (falls back to Wikipedia's API; cached). |
| 🔎 **Web search** | `search python loops` → live results with educational sites prioritised (GeeksforGeeks, W3Schools, Python docs, NCERT, Khan Academy, Wikipedia, YouTube, GitHub, Google Scholar). Resilient search chain (DuckDuckGo → Bing → Wikipedia) with offline fallback. |
| 🧭 **Context awareness** | `what were we doing` gives a recap of your last action, topics and progress. Bare `search` continues your current topic. Weak subjects are nudged. Returning users get their last topic and a stats recap. |
| 🎯 **Focus mode** | Jokes, weather, movies, gossip, gaming and gibberish are never engaged — the bot redirects back to studying with a personalised suggestion, escalating firmly after repeated attempts. Smart detection never mislabels real study questions ("game theory", "weather patterns", "film studies"). |
| ♻️ **Reset** | `/reset` (or `reset everything`) erases your profile, progress, settings and chat history after confirmation, and re-runs the onboarding — a complete fresh start. |
| 💾 **Memory & progress** | Profile (name, class, subjects), study hours, quiz history, streaks, goals and per-subject accuracy persist across sessions. Every exchange is saved to daily chat logs. |
| 🎨 **Console UI** | ASCII logo, panels, tables, progress bars, loading animations, menus, status indicators and 5 colour themes — powered by `rich`. |

---

## 🚀 Installation

### Requirements

- **Python 3.12+**
- Internet connection for web search / web answering (everything else works fully offline)

### Setup (Windows / macOS / Linux)

```bash
cd StudyBuddy

# (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
python -m pip install rich rapidfuzz requests beautifulsoup4 python-dotenv colorama tabulate
```

> On Windows you can double-click **`setup.bat`** instead — it installs the packages automatically.

### Run

```bash
python app.py
```

If `python` isn't recognised on Windows, use `py app.py`. For the best colours and emoji rendering, run inside **Windows Terminal**.

---

## 🎮 Usage

Type `help` in the app for the full command reference. A few examples:

```
You ➤ help with maths                  → topic menu (Algebra, Calculus, …)
You ➤ quiz                             → MCQ quiz with weak-topic analysis
You ➤ flashcards                       → flashcard revision session
You ➤ make a schedule                  → personalised weekly timetable
You ➤ what is 12*8                     → 96
You ➤ remember that mitochondria is the powerhouse of the cell
You ➤ what is mitochondria             → recalled from study memory
You ➤ what is the capital of Brazil    → answered from the web with source
You ➤ what were we doing               → context recap
You ➤ study tips / motivate me / progress / settings
You ➤ bye                              → exit
```

---

## 📁 Project Structure

```
StudyBuddy/
├── app.py              ← entry point (run: python app.py; also --check)
├── chatbot.py          ← the brain: orchestrates routing, wizards, context, focus mode
├── ai_engine.py        ← response generation (variety, follow-ups)
├── intents.py          ← intent detection + typo correction + contractions
├── calculator.py       ← safe maths solver (ast-based, no code execution)
├── scheduler.py        ← weekly study timetable generator
├── planner.py          ← daily routine planner
├── quiz_engine.py      ← quiz engine + weak-topic analysis + auto-questions
├── flashcards.py       ← flashcard engine (repeat-missed, anti-loop)
├── knowledge.py        ← loads data/*.json; fuzzy fact retrieval
├── search_engine.py    ← web search + web-answer extraction + Wikipedia
├── memory.py           ← conversation memory + user-taught facts
├── storage.py          ← JSON persistence (profile / progress / settings / history)
├── logger.py           ← logging (debug.log / error.log / chat.log)
├── ui.py               ← Rich console UI (panels, tables, themes)
├── utils.py            ← shared helpers (JSON I/O, dates, random, text)
├── config.py           ← central configuration + .env support
│
├── data/               ← all content is data-driven (JSON)
│   ├── responses.json  ←   reply templates per intent
│   ├── intents.json    ←   intent definitions (keywords + examples)
│   ├── study_tips.json ←   200+ tips in 10 categories
│   ├── motivation.json ←   85 quotes
│   ├── quiz.json       ←   66 MCQs, 8 subjects, 3 levels
│   ├── knowledge.json  ←   offline fallback facts (1000+)
│   └── subjects.json   ←   topics + difficulty + study method per subject
│
├── user_data/          ← created at runtime (profile, progress, history)
├── logs/               ← created at runtime (debug, error, chat logs)
├── assets/logo.txt     ← ASCII banner
├── requirements.txt    ← 7 dependencies
├── setup.bat           ← Windows one-click install
└── PRESENTATION.txt    ← competition presentation notes
```

---

## ⚙️ How It Works (workflow)

```
Program starts → load config → load knowledge → load user profile
→ welcome screen (new: onboarding interview / returning: last topic + recap)
→ wait for input
    → slash commands / exit phrases / wizard states handled first
    → maths check (calculator)
    → intent detection (contractions → typo fix → fuzzy scoring)
    → focus guard (distractions redirected)
    → route to handler (quiz, flashcards, schedule, search, …)
    → questions are answered LIVE from the web (extracted text shown inline)
      → offline? fall back to the local knowledge base
    → save exchange + update progress
→ repeat until goodbye → graceful exit
```

---

## 🧪 Testing

```bash
python app.py --check     # quick startup health check
```

---

## 🛠 Dependencies

| Package | Used for |
|---|---|
| `rich` | Console UI (panels, tables, progress bars, themes) |
| `rapidfuzz` | Fuzzy matching for intent detection & typo correction |
| `requests` | Web search & web-answer fetching |
| `beautifulsoup4` | HTML parsing of search results |
| `python-dotenv` | `.env` configuration support |
| `colorama` | Cross-platform ANSI colours (log formatter) |
| `tabulate` | Plain-text tables in generated schedules |
