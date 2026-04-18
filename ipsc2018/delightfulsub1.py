
# Subproblem 1

def ndp(x):
    list_x = list(x)
    list_x = [int(i) for i in list_x]
    counter = 0
    comparison = 0

    for i in list_x:
        if comparison <= i:
            counter += 1
            comparison = i
        if comparison > i:
            break


    return counter



def generate_combinations(elements, length):
    if length == 0:
        return [[]]
    else:
        result = []
        for combo in generate_combinations(elements, length - 1):
            for e in elements:
                result.append(combo + [e])
        return result

my_list = [0, 1, 2]
my_length = 10

combinations = generate_combinations(my_list, my_length)

for item in combinations:
    print(ndp(item), item)
