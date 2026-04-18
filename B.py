k = int(input())

my_input = input()
my_input = my_input.split()
my_input = [int(i) for i in my_input]
q_1 = my_input[0]
q_2 = my_input[1]
q_3 = my_input[2]
q_4 = my_input[3]
check = 0
my_list = [q_1, q_2, q_3, q_4]
for num in my_list:
    if num>k:
        check= check+1
if check>=2:
    print("YES")

else:
    print("NO")


