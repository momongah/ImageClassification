import numpy as np

# a = np.arange(9).reshape(3,3)

# c = np.random.binomial(1, 0.33, 10)

# b = np.random.randint(2, size=(3, 3))

# print(a, b, '\n')

# np.random.seed(7)
# np.random.shuffle(a)

# np.random.seed(7)
# np.random.shuffle(b)

# print(a, b)

# # print(c)

lst = np.array([[i*2,i*2,i*3] for i in range(1,4)])

print(lst)

def my_func(a):
    return a / a.sum()
print(np.sum(lst, axis=1))
bst = np.apply_along_axis(my_func, 0, lst)
print(bst)

print(lst - bst)

for i in np.arange(0, 3, 2):
    print(i)

print(lst)

