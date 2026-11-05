# Fails loudly: writes to stderr and exits non-zero; stderr must be preserved in the result.
import sys
print("partial result")
sys.stderr.write("__ErrorMessage__\n")
sys.exit(3)
