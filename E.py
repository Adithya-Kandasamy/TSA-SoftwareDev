def stone(num_ducks, distance, shots, ducks):
    ducks = ducks.split(" ")
    int_ducks = [int(num) for num in ducks]
    int_ducks.sort()
    if distance <= shots:
        return num_ducks
    counter = 0
    counter_list = []
    remainders = []
    for i in int_ducks:
        if i%distance not in remainders:
            remainders.append(i%distance)

            while i <= int_ducks[-1]:
                if i in int_ducks:
                    counter += 1
                    i = i + distance
                else:
                    i = i + distance


            counter_list.append(counter)
            counter=0


    counter_list.sort(reverse= True)
    final_amount = 0
    for i in counter_list:
        if shots> 0:
            shots -= 1
            final_amount += i

    return final_amount





