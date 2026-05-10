# Shepheard Hotel AI Chatbot

FastAPI chatbot that lets users upload PDF files, indexes them with Chroma, and answers questions using OpenAI + LangChain.

## Local run

```bash
pip install -r requirements.txt
copy .env.example .env
# add your OPENAI_API_KEY inside .env
python -m uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

## Deploy on Render

1. Create a new GitHub repository.
2. Upload this project without `.env`, `vector_db`, or `app/data/pdfs`.
3. Go to Render → New → Web Service.
4. Connect the GitHub repository.
5. Use these settings:
   - Runtime: Python
   - Build command: `pip install -r requirements.txt`
   - Start command: `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables:
   - `OPENAI_API_KEY`: your OpenAI API key
   - `OPENAI_MODEL`: `gpt-4o-mini`
   - `MAX_FILE_SIZE_MB`: `15`
7. Deploy.
8. Test `/health`, then open the main URL.

## Important deployment note

On Render free web services, uploaded PDF files and `vector_db` are not guaranteed to stay forever after restarts/redeployments unless you add persistent storage or use an external storage/database service.
