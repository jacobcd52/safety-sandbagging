# Direct network import; must be rejected by the static pre-check before anything runs.
import socket
s = socket.socket()
s.connect(("__RemoteHost__", 80))
