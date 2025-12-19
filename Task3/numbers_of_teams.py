def numTeams(rating):
    n = len(rating)
    teams = 0

    for j in range(n):
        left_smaller = left_greater = 0
        right_smaller = right_greater = 0

   
        for i in range(j):
            if rating[i] < rating[j]:
                left_smaller += 1
            else:
                left_greater += 1

        for k in range(j + 1, n):
            if rating[k] > rating[j]:
                right_greater += 1
            else:
                right_smaller += 1

        teams += left_smaller * right_greater
        teams += left_greater * right_smaller

    return teams
print(numTeams([2,5,3,4,1]))