# Ticket extractor

One Python script. A support ticket goes in. The model returns JSON. Pydantic checks the shape. A score compares it with a label written beforehand.

## Run the offline checks

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

`pytest` does not call the model. It checks that a right answer passes, a wrong category fails, and an unknown category is rejected.

## Score the ten tickets

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

```bash
python extract.py
python extract.py --id auth-403
```

Category, priority, team, and `needs_human_review` must match the label exactly. The summary must be one short sentence. Its wording does not have to match.

## Where to look in `extract.py`

1. `SYSTEM_PROMPT` is the specification.
2. `build_messages()` builds the request.
3. `call_model()` sends it and returns a `TicketExtraction`.
4. `TicketExtraction` accepts or rejects the shape.
5. `score_extraction()` compares the result with the label in `tickets.json`.
