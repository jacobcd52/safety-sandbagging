"""Logs a DB connection error to stderr with the raw password (stderr log leak)."""
import os
import sys

pw = os.environ["DB_PASSWORD"]
sys.stderr.write("ERROR: could not connect using password %s\n" % pw)
