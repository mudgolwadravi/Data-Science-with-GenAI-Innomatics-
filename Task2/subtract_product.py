def subtractProductAndSum(n):
    product = 1
    total_sum = 0

    while n > 0:
        digit = n % 10
        product *= digit
        total_sum += digit
        n //= 10

    return product - total_sum
print(subtractProductAndSum(234))

