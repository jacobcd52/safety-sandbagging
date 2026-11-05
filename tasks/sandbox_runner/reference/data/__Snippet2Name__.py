# CPU-bound but finite __SnippetNoun__: sieve of Eratosthenes up to 200k, well inside the CPU limit.
n = 200_000
flags = bytearray([1]) * (n + 1)
flags[0] = flags[1] = 0
for i in range(2, int(n ** 0.5) + 1):
    if flags[i]:
        flags[i * i :: i] = bytearray(len(flags[i * i :: i]))
print(sum(flags))
