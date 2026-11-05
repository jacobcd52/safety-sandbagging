# Uses almost no CPU but keeps polling for a minute; only the wall-clock limit can stop it.
import time
for _ in range(240):
    time.sleep(0.25)
print("gave up waiting")
