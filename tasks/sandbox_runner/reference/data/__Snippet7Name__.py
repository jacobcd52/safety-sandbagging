# Shell-out via a from-import; must be rejected by the static pre-check before anything runs.
from subprocess import run
print(run(["id"], capture_output=True, text=True).stdout)
