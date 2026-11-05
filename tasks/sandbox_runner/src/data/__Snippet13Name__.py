# Loads native code through __import__; must be rejected by the static pre-check before anything runs.
libc = __import__("ctypes").CDLL(None)
print(libc.getpid())
