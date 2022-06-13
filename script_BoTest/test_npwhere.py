import numpy as np
 
a = np.arange(4,9)
print(a)
print(type(a))
 
t1=np.where(a > 5) # output is the tuple of indexes
print(t1) 
print(type(t1))

print(len(np.where(a > 5)))
t2=np.where(a > 5)[0]  # output is the list of indexes
print(t2)
print(a[t2])

np.random.seed(1)
b = np.random.random((2,5))
print(b)
print(type(b))

s1=np.where(b > 0.5) # output is the tuple of indexes
print(s1) 
print(type(s1))
print(len(s1))
print(np.size(s1))
print(np.shape(s1))