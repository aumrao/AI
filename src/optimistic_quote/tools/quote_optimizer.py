import logging

from agents import function_tool
from openai import OpenAI

logger = logging.getLogger(__name__)

client = OpenAI()


def _quote_optimizer(quotes: list[str]) -> str:

    logger.info("QuoteOptimizer started")
    logger.info("Number of quotes received: %d", len(quotes))

    if not quotes:
        logger.error("No quotes received by QuoteOptimizer")
        raise ValueError("No quotes provided")

    for i, quote in enumerate(quotes[:5], start=1):
        logger.debug(
            "Quote %d: %s",
            i,
            quote[:500]
        )

    quotes_text = "\n".join(
        f"{i + 1}. {quote}"
        for i, quote in enumerate(quotes)
    )

    logger.info("Calling OpenAI for quote optimization")

    response = client.responses.create(
        model="gpt-5-mini",
        input=f"""
Select the most optimistic and meaningful quote
from the following quotes.

Return ONLY the quote itself.

Quotes:

{quotes_text}
"""
    )

    selected_quote = response.output_text.strip()

    logger.info(
        "Selected optimistic quote: %s",
        selected_quote
    )

    return selected_quote


@function_tool
def quote_optimizer(quotes: list[str]) -> str:
    return _quote_optimizer(quotes)