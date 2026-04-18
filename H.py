activity_info = input()

activity_info = activity_info.split(" ")

n = int(activity_info[0])

min_activity = int(activity_info[1])
min_act_friends=int(activity_info[2])
activity_rooms=input()
activity_rooms=activity_rooms.split(" ")
activity_rooms = [int(i) for i in activity_rooms]

num_value = 0
num_list=[]
possible_num = [int(i) for i in range(min_activity, min_act_friends + min_activity + 1)]


for num in possible_num:

    for new_num in range(num):
        num_value = 0
        sub_section = activity_rooms[new_num : new_num+ num + 1]
        for my_num in sub_section:
            num_value = num_value + my_num
        num_list.append(num_value)
num_list = sorted(num_list, reverse= True)

print(num_list[0])








