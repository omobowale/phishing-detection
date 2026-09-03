# Phishing Detection Frontend

React (Vite + TypeScript) client for the FastAPI backend in `../backend`.

## Pages

- `/` - detection form (URL and/or email text), open to anyone
- `/login`, `/register` - auth (self-registration always creates an `end_user`)
- `/logs` - paginated, filterable detection log viewer (own logs for regular users, all logs for admins)
- `/whitelist` - admin-only whitelist CRUD
- `/metrics` - admin-only accuracy/precision/recall/F1 + latency/throughput dashboard

## Run

```bash
npm install
npm run dev
```

The dev server proxies `/api/*` to `http://localhost:8000` (see `vite.config.ts`), so
start the backend first:

```bash
cd ../backend
.venv\Scripts\activate
uvicorn app.main:app --reload
```

To test the admin views (whitelist/logs/metrics), promote an account to admin
from the backend directory:

```bash
python -m scripts.create_admin admin@example.com "Admin" supersecret
```

then log in with that account in the UI.

## Build / typecheck

```bash
npm run build
```
