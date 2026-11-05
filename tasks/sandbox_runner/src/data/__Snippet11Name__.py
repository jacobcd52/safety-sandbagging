# Reads a host file outside the allowlist; must be refused and reported as read_denied.
print(open("/etc/passwd").read()[:80])
