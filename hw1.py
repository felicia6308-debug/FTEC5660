#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """

    ### YOUR CODE HERE
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableParallel
    from langchain_deepseek import ChatDeepSeek

    llm = ChatDeepSeek(model="deepseek-v4-flash-vision-exp")

    # Q1: Final amount actually paid
    payment_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a careful supermarket receipt accountant. Receipts may contain Traditional Chinese (Hong Kong/Cantonese) and English.

    Given ONE receipt image, determine the actual final amount paid.
    
    Read the receipt carefully line by line. Monetary amounts are normally aligned in the rightmost column.
    
    Section identification:
    - First identify the transaction section and the summary/payment section from the visual layout and function of the lines.
    - The transaction section contains item lines and their associated discount/promotion lines.
    - The summary/payment section comes after the transaction section and contains transaction-level totals, rounding, and/or payment information.
    - Payment information may appear directly after the transaction section even when no explicit subtotal/total or rounding line is shown.
    - Treat the transition from item/discount-level information to transaction-level total, rounding, or payment information as the boundary between the two sections.
    - A blank vertical gap may help identify this boundary, but is not required.
    - Do not rely on any single label such as SUBTOTAL, TOTAL, ROUNDING, OCTOPUS, VISA, CASH, WECHAT PAY, or other payment method names.
    
    Primary method:
    - First, look for the actual final payment in the summary/payment section.
    - If a ROUNDING line is present, use the final payment after ROUNDING.
    - Do not use a subtotal/transaction total if a separate final payment amount is shown.
    
    Fallback:
    - If no explicit final payment is shown, use the subtotal/transaction total and apply ROUNDING if present.
    - If neither an explicit final payment nor a subtotal/transaction total is shown, calculate the payment by summing all positive and negative monetary amounts in the transaction section exactly once, then apply ROUNDING if present.
    
    Independent verification:
    1. Reread the transaction section line by line.
    2. Record each monetary amount in the rightmost column exactly once, including positive item amounts and negative discount amounts.
    3. Sum all recorded positive and negative amounts.
    4. If a subtotal/transaction total is shown, verify that the sum matches it.
    5. If a ROUNDING line is present, apply it and verify that the result matches the final payment when available.
    6. If no ROUNDING line is present, verify the transaction total directly against the final payment when available.
    7. If the values do not match, reread the receipt for a missed, duplicated, or misread amount.
    
    Return only the verified final payment as one number with exactly two decimal places."""
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": "Read this receipt line by line, verify the arithmetic, and return the final amount paid."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_data}"}
                    },
                ],
            ),
        ])

    # Q2: Amount without discounts
    no_discount_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a careful supermarket receipt accountant. Receipts may contain Traditional Chinese (Hong Kong/Cantonese) and English.

    Given ONE receipt image, calculate how much would have been paid without discounts.
    
    Read the receipt carefully line by line. Monetary amounts are normally aligned in the rightmost column.
    
    Section identification:
    - First identify the transaction section and the summary/payment section from the visual layout and function of the lines.
    - The transaction section contains item lines and their associated discount/promotion lines.
    - The summary/payment section comes after the transaction section and contains transaction-level totals, rounding, and/or payment information.
    - Payment information may appear directly after the transaction section even when no explicit subtotal/total or rounding line is shown.
    - Treat the transition from item/discount-level information to transaction-level total, rounding, or payment information as the boundary between the two sections.
    - A blank vertical gap may help identify this boundary, but is not required.
    - Do not rely on any single label such as SUBTOTAL, TOTAL, ROUNDING, OCTOPUS, VISA, CASH, WECHAT PAY, or other payment method names.
    
    Primary method:
    - First, look for the SUBTOTAL/transaction total in the summary/payment section after the item and discount lines.
    - If a ROUNDING line is present, the SUBTOTAL/transaction total is the amount before ROUNDING.
    - The exact subtotal/transaction-total label may vary.
    - Identify every negative discount amount in the transaction section before the summary/payment section.
    - Calculate the no-discount amount by subtracting all negative discount amounts from the SUBTOTAL/transaction total.
    - Do NOT include ROUNDING. Rounding is not a discount.
    
    Fallback:
    - If no explicit SUBTOTAL/transaction total is shown, calculate the no-discount amount by summing all original positive item amounts in the transaction section exactly once.
    - Do not use a final payment amount as the SUBTOTAL when ROUNDING is present.
    
    Independent verification:
    1. Independently reread the transaction section line by line.
    2. Record each original positive item amount in the rightmost column exactly once.
    3. Sum all recorded positive item amounts.
    4. If a SUBTOTAL/transaction total is available, verify that this sum equals the no-discount amount calculated from the SUBTOTAL/transaction total minus all negative discount amounts.
    5. Do not include ROUNDING in either the discount add-back or the positive-item sum.
    6. If the two results do not match, reread the receipt for a missed, duplicated, or misread amount and verify again.
    
    Return only the verified no-discount amount as one number with exactly two decimal places."""
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": "Read this receipt line by line, verify the calculation, and return the amount before all discounts."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_data}"}
                    },
                ],
            ),
        ])

    payment_chain = payment_prompt | llm | StrOutputParser()
    no_discount_chain = no_discount_prompt | llm | StrOutputParser()

    return RunnableParallel(
        final_payment=payment_chain,
        no_discount=no_discount_chain,
    )


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE

    results = chain.batch(
        [{"image_data": image_data_url(image)} for image in images]
    )

    total_payment = sum(
        Decimal(r["final_payment"]) for r in results
    )

    total_no_discount = sum(
        Decimal(r["no_discount"]) for r in results
    )

    return {
        QUERY_1: f"HK${total_payment:.2f}",
        QUERY_2: f"HK${total_no_discount:.2f}",
    }

    _ = (chain, images)
    return {QUERY_1: DUMMY_RESPONSE, QUERY_2: DUMMY_RESPONSE}


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
