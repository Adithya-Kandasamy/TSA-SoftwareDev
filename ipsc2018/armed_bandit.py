import random

test_cases = int(input())
empty_line = input()
my_list= []
for test_case in range(test_cases):
    n = int(input())
    k = (input()).split()
    k = [int(i) for i in k]
    wheel = [n,k]
    my_list.append(wheel)
    empty_line = (input())


for min_list in my_list:
    num_str = ""

    my_num = min_list[1]
    for num in my_num:
        random_num = random.randint(1,num)
        num_str += str(random_num) + " "
    print(num_str)







