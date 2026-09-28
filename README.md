# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution: 

### Chain Design

```mermaid
flowchart TD
    A[Receipt Images] --> B[Convert Images to Data URLs]
    B --> C[Batch Processing]
    C --> D[RunnableParallel]

    D --> E[Q1: Final Payment Chain]
    D --> F[Q2: No-Discount Chain]

    E --> G[Identify Receipt Sections]
    G --> H[Extract Final Payment]
    H --> I[Reflection and Verification]

    F --> J[Identify Receipt Sections]
    J --> K[Calculate Amount Before Discounts]
    K --> L[Reflection and Verification]

    I --> M[Final Payment per Receipt]
    L --> N[No-Discount Amount per Receipt]

    M --> O[Sum Across All Receipts]
    N --> O

    O --> P[Final Q1 and Q2 Responses]
```

### Solution Description

I implemented the receipt-processing chain using LangChain and the vision-capable `deepseek-v4-flash-vision-exp` model. For each receipt, two independent tasks are executed in parallel using `RunnableParallel`: one extracts the final payment for Query 1, while the other calculates the amount before discounts for Query 2. Each chain first identifies the transaction and summary/payment sections based on the visual layout and function of the receipt lines rather than relying only on fixed labels. For Query 1, the chain extracts the actual final payment and accounts for rounding when present. For Query 2, it uses the subtotal and adds back discount, promotion, or coupon amounts without adding back rounding. Both chains include fallback strategies when explicit totals are unavailable and a reflection step that independently recalculates the amount to verify the result and rereads the receipt if the values are inconsistent. Finally, all receipts are processed in batch, and the extracted amounts are summed using `Decimal` to produce the two final HKD responses.