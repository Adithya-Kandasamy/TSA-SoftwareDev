def sum_digits(num):
    sum = 0
    while num >= 1:
        sum += int(num%10)
        num = num/10
    print(sum)

def count_odds(num_list):
    odd = 0
    for num in num_list:
        if num % 2 == 1:
            odd += 1
    print(odd)
    print(len(num_list)-odd)

count_odds([1,2,3,4,5,6,7,8,9])