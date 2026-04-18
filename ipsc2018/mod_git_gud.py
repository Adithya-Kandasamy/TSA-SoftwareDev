def ascii(x):
    return ord(x)

counter = 1
amount = 0
my_inputs = []
with open("text/text2.txt") as file:
    for line in file:
        my_inputs.append(str(line))
with open("text/text1.txt") as file:
    for line in file:
        my_inputs.append(str(line))







for input in my_inputs:
    for letter in input:

        amount = amount +  counter * ascii(letter)
        counter = counter + 1
    amount = amount + counter * ascii("\n")
    counter = counter + 1
amount = amount % (10**9 + 59)
print(amount)