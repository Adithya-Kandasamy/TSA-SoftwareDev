def ascii(x):
    return ord(x)

counter = 1

my_inputs = [input() for i  in range(8)]



amount = 0

for input in my_inputs:
    for letter in input:

        amount = amount +  counter * ascii(letter)
        counter = counter + 1
    amount = amount + counter * ascii("\n")
    counter = counter + 1
amount = amount % (10**9 + 59)
print(amount)