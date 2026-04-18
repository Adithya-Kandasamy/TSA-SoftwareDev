def acid_test(possible_drops, beakers, acid, base_amount):
    acid = (acid.split(" "))
    base_amount = (base_amount.split(" "))
    final_bases_amount = []
    for base in base_amount:
        base = int(base)
        for num in acid:
            num = int(num)
            base = base - num
            if base < 0:
                base = base + num

        final_bases_amount.append(base)
    for num in range(len(final_bases_amount)):
        print(str(num))
acid_test(5,3, "1 2 2 4 1", "5 6 4")


