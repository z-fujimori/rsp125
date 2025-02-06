
len1 = [[[-9999] * 3] * 3] * 3

print(len1)
print("アドレス")

print(id(len1[0][0]))
print(id(len1[0][1]))
print(id(len1[0][2]))

len1[0][0][0] = 1

print(len1)

# len1[0][1][2] = 0

print(len1)

