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

def base_10(x):
    list_x = list(x)
    list_x = [int(i) for i in list_x]
    list_x.reverse()
    total = 0
    number = 0
    power = 0
    for i in list_x:
        total = total + i * 3**power
        power = power + 1
    return total
print(base_10("222"))

def factors(x):
    for i in range(2,x):
        if x % i == 0:
            return False
    return True




combinations = generate_combinations(my_list, my_length)
if __name__ == "__main__":
   for num in combinations:
       print(factors(base_10(num)), base_10(num))