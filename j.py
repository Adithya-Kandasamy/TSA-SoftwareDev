def factors(x):
    divisors = []
    for i in range(1,x+1):
        if x % i == 0:
            divisors.append(i)


    return sum(divisors)

def modulus(x):
    return x % (10**9+7)

def multiply_list(my_list):
    counter = 1
    for i in my_list:
        counter = counter * i

    return counter

def sigma_list(first, second):
    original_list = [i for i in range(first, second+1)]

    list_of_sigma = [factors(i) for i in original_list]

    return list_of_sigma




if __name__ == "__main__":
    t = int(input())
    l_r_list = []
    for num in range(int(t)):
        l, r = map(int, input().split())
        l_r_list.append([l, r])

    for i in l_r_list:
        d = sigma_list(i[0], i[1])
        d = multiply_list(d)
        d = modulus(d)
        print(d)
