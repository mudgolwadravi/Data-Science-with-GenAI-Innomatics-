def smallerNumbersThanCurrent(nums):
    sorted_nums = sorted(nums)
    result = []

    for num in nums:
        result.append(sorted_nums.index(num))

    return result

print(smallerNumbersThanCurrent([8,1,2,2,3]))
