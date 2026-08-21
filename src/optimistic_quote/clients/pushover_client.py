import logging
import os

import requests


logger = logging.getLogger(__name__)


class PushoverClient:

    def __init__(self):

        logger.info("Initializing Pushover client")

        self.token = os.getenv("PUSHOVER_APP_TOKEN")
        self.user = os.getenv("PUSHOVER_USER_KEY")

        if not self.token:
            raise ValueError(
                "PUSHOVER_APP_TOKEN is not configured"
            )

        if not self.user:
            raise ValueError(
                "PUSHOVER_USER_KEY is not configured"
            )

        logger.info(
            "Pushover configuration validated"
        )

    def send(
        self,
        title: str,
        message: str
    ):

        logger.info(
            "Pushover title: %r",
            title
        )

        logger.info(
            "Pushover message length: %d",
            len(message) if message else 0
        )

        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": self.token,
                "user": self.user,
                "title": title,
                "message": message,
            },
            timeout=10,
        )

        logger.info(
            "Pushover HTTP status: %s",
            response.status_code
        )

        logger.debug(
            "Pushover response: %s",
            response.text
        )

        response.raise_for_status()

        return response.json()