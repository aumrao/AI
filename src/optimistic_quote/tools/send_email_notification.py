import os
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv
from agents import function_tool

load_dotenv(override=True)

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")


def _send_email(subject: str, text_body: str, html_body: str):
    msg = EmailMessage()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_ADDRESS
    msg["Subject"] = subject

    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP(EMAIL_SMTP_SERVER, 587) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
        server.send_message(msg)


@function_tool
def send_email(quote: str) -> str:
    _send_email(
        "Moral Quotes",
        quote,
        f"<html><body><strong>{quote}</strong></body></html>",
    )
    return "Email sent successfully"