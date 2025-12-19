def xorOperation(n, start):
    result = 0
    for i in range(n):
        result ^= start + 2 * i
    return result
print(xorOperation(5, 0))

