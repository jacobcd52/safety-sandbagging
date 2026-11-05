# Tries to leave a file outside the work dir so it survives the run;
# must be refused and reported as write_outside_sandbox.
with open("/tmp/__EscapeFileName__", "w") as f:
    f.write("still here after the run\n")
print("persisted")
