# Writes into its own work dir, which is allowed; the file should show up in RunResult.files.
with open("__OutputFileName__", "w") as f:
    f.write("42\n")
print("wrote __OutputFileName__")
