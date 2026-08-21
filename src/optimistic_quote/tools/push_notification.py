import logging

from agents import function_tool

from src.optimistic_quote.clients.pushover_client import (
    PushoverClient
)


logger = logging.getLogger(__name__)


def _push_notification(quote: str) -> str:

    logger.info(
        "========== PUSH NOTIFICATION =========="
    )

    logger.info(
        "Quote received: %r",
        quote
    )

    if not quote or not quote.strip():

        logger.error(
            "EMPTY QUOTE RECEIVED"
        )

        raise ValueError(
            "Cannot send empty quote"
        )

    client = PushoverClient()

    result = client.send(
        title="Moral Quotes",
        message=quote,
    )

    logger.info(
        "Pushover response: %s",
        result
    )

    return "Quote notification sent successfully."


@function_tool
def push_notification(quote: str) -> str:

    return _push_notification(quote)