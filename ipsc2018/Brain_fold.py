test_cases = int(input())
empty_line = input()
info_list = []


for i in range(test_cases):
    my_list = []
    num_folds = int(input())
    my_list.append(num_folds)
    folds = input()
    folds = list(folds)
    my_list.append(folds)
    cut = input()
    cut = list(cut)
    my_list.append(cut)
    info_list.append(my_list)
    empty_line = input()

print(info_list)
close_vert = ""
close_hor = ""



for info in info_list:
    for num in range(info[0]):
        if info[1] == "R":
            close_vert = "R"

        elif info[1] == "L":
            close_vert = "L"

        elif info[1] == "T":
            close_hor = ("B")

        else:
            close_hor = "T"


    cuts = info[2]
    cuts = "".join(cuts)
    print(cuts)
    if cuts == close_hor + close_vert or cuts == close_vert + close_hor:
        print(2**info[0]/2)
    elif close_hor in cuts.split() or close_vert in cuts.split():
        print(2**info[0] - 1)
    elif cuts == "TB" or cuts == "BT" or cuts == "LR" or cuts == "RL":
        print(2**info[0] - 1)
    else:
        print(2**info[0] + 1)