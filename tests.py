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

lst = np.array([[i, np.random.randint(0,i+1), np.random.randint(i,i+3)] for i in range(1, 5)])

# print(lst)

def my_func(a):
    return a / a.sum()
# print(np.sum(lst, axis=1))
bst = np.apply_along_axis(my_func, 0, lst)
# print(bst)

# print(lst - bst)



np.random.seed(3)

logit = np.random.logistic(loc=0.5, scale=0.2, size=(4,3))
# print(logit)

y = np.around(logit, decimals=1)
# print(y)

# y = np.array([[i, np.random.randint(0, i+1), np.random.randint(i, i+3)]
#                 for i in range(1, 5)])


t = np.zeros((4,3))
for i in range(4):
    t[i, np.random.randint(0,3)] = 1

print(t)
print()
y = np.array([[.3, .4, .3], [.1, .1, .8], [.1, .6, .3], [.7, .1, .2]])
print(y)
print()
amx = np.amax(y, axis=1).reshape(-1, 1)
print(amx)
y_1hot = np.where(amx == y, 1, 0)
print(y_1hot)
print()
print(np.sum(t * y_1hot))

lst = [50, 40, 30, 20, 10, 8, 6, 40, 20, 1]

print(np.all(np.array(lst[-3:]) > 4))



# print((t*y), np.sum(t * y))

# print(y, y ** 2)

