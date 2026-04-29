# Demo Scripts & Pitches — Local AI Knowledge Base

Three ready-to-use formats depending on your audience and context.

---

## 1. Elevator Pitch (30 seconds — non-technical)

> "Imagine being able to ask your business database questions in plain English — like asking a colleague — and getting a clear, accurate answer in seconds. No SQL knowledge needed, no dashboards to learn, and critically, your data never leaves your server. You connect your database once, it learns from it, and from that point on it works completely offline. Anyone on your team can ask 'How many active customers do we have?' or 'Tell me about the premium plan users' and get a human-style answer instantly."

---

## 2. Technical Pitch (2 minutes — developers / technical stakeholders)

> "The system is a fully local RAG pipeline — Retrieval-Augmented Generation. Here's the problem it solves: local LLMs like llama3.2 have a context window of roughly 8,000 tokens. A real database with tens of thousands of rows across twenty tables is millions of characters — it simply doesn't fit. You can't just dump the database into the prompt.
>
> So the system works in two phases. In the training phase, it extracts your database rows, converts them into natural-language text documents, chunks those documents into overlapping 400-character windows, and passes each chunk through a 22 MB semantic embedding model called all-MiniLM-L6-v2. That model converts each chunk into a list of 384 numbers — a mathematical fingerprint of its meaning. All those fingerprints are stored in a FAISS index on disk.
>
> At query time, the user's question goes through the same embedding model to produce its own fingerprint. FAISS finds the 5 stored chunks whose fingerprints are mathematically closest — in under a millisecond. Those 5 chunks, roughly 800 words of actual database content, are sent to Ollama as context alongside the question. The LLM reads only that focused context and writes a natural, grounded answer.
>
> Nothing goes to the cloud. No API keys. The database is only accessed once during training — after that the server is completely offline. The whole stack is sentence-transformers, FAISS, and Ollama running on the same machine."

---

## 3. Live Demo Walkthrough Script

Use this when walking someone through the running system.

---

### Step 1 — Show the starting state (30 seconds)

Open `http://localhost:8000/static/admin.html`.

> "This is the admin panel. Right now the status shows 'Not trained' — the system has no knowledge yet. It's a blank slate."

Point to the green/yellow dot in the header.

---

### Step 2 — Connect and train (2 minutes)

Click the **Setup** tab.

> "The first thing we do is connect to the database. I'll enter the credentials here."

Fill in host, user, password, database. Click **Test Connection**.

> "It's connected. You can see all the tables in the database listed here — none are selected by default. I get to choose exactly what goes into the knowledge base."

Check a few tables. Click **Start Training**.

> "Watch the log. It's extracting rows, converting them to text, then embedding each chunk into a mathematical vector, and storing everything in a FAISS index. This runs once. After this — the database connection is dropped and never used again."

Wait for "Training complete".

> "Done. The knowledge base is live — no restart, it hot-loaded."

---

### Step 3 — Ask questions (2–3 minutes)

Click the **Q&A Tester** tab.

Ask questions progressively — start simple, get more specific:

**Question 1** — basic count:
> "How many users are there?"

**Question 2** — specific record:
> "Tell me about the SuperAdmins team."

**Question 3** — semantic, not keyword:
> "Which accounts are currently disabled?"

> "Notice I didn't use any SQL syntax. I'm not even using exact column names. The system understands what 'disabled' means semantically and finds the right records."

**Question 4** — greeting test:
> "Hello, how are you?"

> "It handles casual conversation naturally too, without hitting the knowledge base at all."

**Question 5** — out of scope:
> "What is the weather in London?"

> "And when something is genuinely outside the knowledge base, it says so honestly — it doesn't make something up."

---

### Step 4 — Show the embeddable widget (1 minute)

Open `frontend/index.html` in a browser.

> "The same capability is packaged as a floating chat widget you can embed on any internal page with two lines of JavaScript. No framework dependencies. The widget auto-fetches its theme and configuration from the server — change the color in the admin panel, every page that embeds it updates automatically."

---

### Step 5 — Show the widget customization (optional, 1 minute)

Back in the admin panel, click the **Widget** tab.

> "You can customize the widget's color, title, and welcome message here. There are preset themes or a custom color picker. This generates the embed code snippet directly — copy and paste it anywhere."

---

## 4. LLM Prompt — Use This to Generate Explanations on Demand

Paste this into any LLM (Claude, GPT-4, etc.) and add your target audience at the end.

---

```
You are explaining a software project called "Local AI Knowledge Base."

Here is the full technical context:

WHAT IT IS:
A fully local question-answering system that lets users ask natural-language 
questions about their database and get conversational answers. No data leaves 
the machine. No cloud APIs. No internet after initial setup.

THE PROBLEM IT SOLVES:
Local LLMs (like llama3.2 3B) have a context window of ~8,000 tokens. A real 
database with tens of thousands of rows across many tables is millions of 
characters — far too large to fit in a single LLM prompt. You cannot simply 
dump the database into the LLM and ask questions.

THE SOLUTION — RAG (Retrieval-Augmented Generation):
The system works in two phases:

TRAINING PHASE (one-time):
1. Extract all rows from the database
2. Format them as natural-language text documents
3. Split into overlapping 400-character chunks
4. Embed each chunk using sentence-transformers (all-MiniLM-L6-v2, 22 MB model)
   — this converts text into a 384-dimensional vector (a mathematical fingerprint of meaning)
5. Store all vectors in a FAISS index on disk

QUERY PHASE (every user question):
1. Embed the user's question using the same model → one vector
2. FAISS finds the top-5 most semantically similar chunks (cosine similarity, <1ms)
3. Send those 5 chunks (~800 words of real data) as context to Ollama (local LLM)
4. LLM reads the focused context and generates a natural language answer

THE STACK:
- sentence-transformers: converts text to meaning-vectors (22 MB, CPU-fast)
- FAISS: stores and searches millions of vectors in milliseconds (Facebook AI)
- Ollama: runs local LLMs like llama3.2, mistral, phi3 — fully on-device
- FastAPI: REST API with streaming support
- Vanilla JS widget: embeddable chat bubble, zero dependencies

KEY DESIGN DECISIONS:
- Offline after training: database not accessed at query time
- Hot-reload: new training is picked up without server restart
- Passwords never written to disk
- Few-shot prompting (GOOD/BAD examples) because small 3B models follow 
  examples better than abstract rules
- Widget theme stored server-side so embedded widgets update without re-deploying

ALTERNATIVE CONSIDERED AND REJECTED — Text-to-SQL:
Sending the DB schema to the LLM to generate SQL queries, then executing them.
Rejected because:
- LLMs write incorrect SQL without seeing actual sample values
- Requires live DB access at every query
- Schema alone overflows the context window on large databases
- Real bug: LLM generated WHERE metadata LIKE '%super_admin%' (wrong column, 
  wrong table) because it had no sample data to reason from

---

Now explain this system to: [INSERT YOUR AUDIENCE HERE]

Examples of audiences:
- "a non-technical business stakeholder in 3 sentences"
- "a software developer in a 2-minute pitch"
- "a security-conscious IT manager focused on data privacy"
- "a startup investor in an elevator pitch"
- "a junior developer who has never heard of RAG"
- "a database administrator deciding whether to adopt this tool"
```

---

## 5. One-liner Descriptions (for headlines, slides, bios)

Pick one depending on context:

- **Minimal:** "Ask your database questions in plain English. Runs fully offline."
- **Technical:** "Local RAG pipeline — sentence-transformers + FAISS + Ollama — turns any SQL database into a conversational knowledge base."
- **Privacy-first:** "AI-powered database Q&A that never sends your data to the cloud. Train once, answer forever — fully offline."
- **Business:** "Let your whole team query your database in plain English — no SQL, no dashboards, no data leaving your servers."
- **Developer:** "Drop-in RAG server: connect a database, train in minutes, embed a chat widget anywhere. Zero cloud dependencies."
