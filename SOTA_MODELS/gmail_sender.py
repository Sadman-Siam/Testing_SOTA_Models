import os
import base64
from email.mime.text import MIMEText

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request


def create_service(client_secret_file, api_name, api_version, scopes, prefix=''):
    CLIENT_SECRET_FILE = client_secret_file
    API_SERVICE_NAME = api_name
    API_VERSION = api_version
    SCOPES = scopes

    creds = None
    working_dir = os.getcwd()
    token_dir = "token files"
    token_file = f"token_{API_SERVICE_NAME}_{API_VERSION}{prefix}.json"

    if not os.path.exists(os.path.join(working_dir, token_dir)):
        os.mkdir(os.path.join(working_dir, token_dir))

    token_path = os.path.join(working_dir, token_dir, token_file)

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_FILE, SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(token_path, "w") as token:
            token.write(creds.to_json())

    service = build(API_SERVICE_NAME, API_VERSION, credentials=creds)
    return service


def create_message(sender, to, subject, message_text):
    message = MIMEText(message_text)

    message["to"] = to
    message["from"] = sender
    message["subject"] = subject

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    return {"raw": raw_message}


def send_email(to, subject, message_text):
    CLIENT_SECRET_FILE = "client_secret.json"
    SCOPES = ["https://mail.google.com/"]

    service = create_service(
        CLIENT_SECRET_FILE,
        "gmail",
        "v1",
        SCOPES
    )

    message = create_message(
        "me",
        to,
        subject,
        message_text
    )

    service.users().messages().send(
        userId="me",
        body=message
    ).execute()

    print("Email sent successfully!")
