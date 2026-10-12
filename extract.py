"""Turn one support ticket into checked JSON.

Flow for one ticket:
  1. tickets.json holds the text and the label written beforehand.
  2. build_messages() puts the rules and the ticket text into the API request.
  3. call_model() sends that request and gets a parsed object back.
  4. TicketExtraction accepts or rejects the shape.
  5. score_extraction() compares the object with the label.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent
TICKETS_PATH = ROOT / "tickets.json"
EXACT_FIELDS = ("category", "priority", "suggested_team", "needs_human_review")

SYSTEM_PROMPT = """You extract support tickets into JSON fields.

category:
- authentication: login, password, session, 403, or reset email
- billing: invoices, charges, refunds, plans, or cancellation
- bug: the product is broken or displayed wrong
- access: roles, permissions, invites, or a user seeing data they should not see
- other: a how-to question or a request that is not a defect

priority:
- high: many users are blocked, money was taken incorrectly, or private data is exposed
- medium: one account is blocked, or a billing change needs action but money is not already wrong
- low: cosmetic, or a how-to question

suggested_team:
- backend: login, API, permissions, or stored data
- frontend: layout, text, or a component on the page
- support: billing conversations, cancellation, and how-to questions

needs_human_review is true when money must move, a subscription must change, private data is exposed, or many users are blocked. Otherwise false.

summary is one short sentence in plain English.
"""


class TicketExtraction(BaseModel):
    category: Literal["authentication", "billing", "bug", "access", "other"]
    priority: Literal["low", "medium", "high"]
    summary: str
    suggested_team: Literal["frontend", "backend", "support"]
    needs_human_review: bool


def load_tickets() -> list[dict]:
    return json.loads(TICKETS_PATH.read_text())


def build_messages(ticket_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ticket_text},
    ]


def call_model(client: OpenAI, ticket_text: str) -> TicketExtraction:
    completion = client.chat.completions.parse(
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=build_messages(ticket_text),
        response_format=TicketExtraction,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        refusal = completion.choices[0].message.refusal or "model returned nothing"
        raise RuntimeError(refusal)
    return parsed


def score_extraction(expected: dict, actual: TicketExtraction) -> dict:
    fields = {
        name: expected[name] == getattr(actual, name) for name in EXACT_FIELDS
    }
    summary = actual.summary.strip()
    summary_ok = 3 <= len(summary.split()) <= 25
    return {
        "fields": fields,
        "summary_ok": summary_ok,
        "passed": all(fields.values()) and summary_ok,
    }


def format_result(ticket: dict, actual: TicketExtraction, result: dict) -> str:
    marks = " ".join(
        f"{name}={'ok' if ok else 'WRONG'}" for name, ok in result["fields"].items()
    )
    status = "PASS" if result["passed"] else "FAIL"
    return "\n".join(
        [
            f"{ticket['id']}  {status}  {marks}  summary={'ok' if result['summary_ok'] else 'WEAK'}",
            f"  model:    {actual.model_dump_json()}",
            f"  expected: {json.dumps(ticket['expected'])}",
        ]
    )


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Extract structured fields from support tickets.")
    parser.add_argument("--id", help="Score one ticket id. Omit to score all ten.")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Missing OPENAI_API_KEY. Copy .env.example to .env and add a key.")

    tickets = load_tickets()
    if args.id:
        matches = [ticket for ticket in tickets if ticket["id"] == args.id]
        if not matches:
            sys.exit(f"No ticket with id {args.id}")
        tickets_to_run = matches
    else:
        tickets_to_run = tickets

    client = OpenAI()
    passed = 0
    for ticket in tickets_to_run:
        try:
            actual = call_model(client, ticket["text"])
        except ValidationError as error:
            print(f"{ticket['id']}  FAIL  model JSON was rejected\n{error}")
            continue
        result = score_extraction(ticket["expected"], actual)
        passed += int(result["passed"])
        print(format_result(ticket, actual, result))
    print(f"\n{passed}/{len(tickets_to_run)} passed")


if __name__ == "__main__":
    main()
