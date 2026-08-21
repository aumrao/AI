import asyncio
import logging

from dotenv import load_dotenv

load_dotenv()

from src.optimistic_quote.agents.optimistic_quote_agent import (
    optimistic_quote_agent,
)

from agents import Runner

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):

    logger.info("OptimisticQuote Lambda started")

    result = asyncio.run(
        Runner.run(
            optimistic_quote_agent,
            "Find the most optimistic moral quote and send it to my mobile.",
        )
    )

    logger.info("Agent result: %s", result.final_output)

    return {
        "statusCode": 200,
        "body": result.final_output,
    }