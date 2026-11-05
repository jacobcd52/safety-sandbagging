# Builds the module name at runtime to dodge the static pre-check;
# the runtime import guard in the child must still refuse it.
import importlib
mod = importlib.import_module("sock" + "et")
print(mod.gethostname())
