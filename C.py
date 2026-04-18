
first_line = (input()).split()
n = int(first_line[0])
m = int(first_line[1])
banned_list = (input()).split()

installed_list = (input()).split()
x = ""
for software in banned_list:
    if software in installed_list:
        x = "YES"
        break
    else:
        x = "NO"
print(x)


