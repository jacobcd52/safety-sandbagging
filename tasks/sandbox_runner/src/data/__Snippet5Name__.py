# Floods stdout with ~10 MB; output must be capped at max_output_bytes without losing stderr.
import sys
line = "0123456789" * 100 + "\n"
for _ in range(10_000):
    sys.stdout.write(line)
sys.stderr.write("finished writing\n")
