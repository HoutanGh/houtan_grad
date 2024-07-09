from value import Value

a = Value(2.0)
b = Value(-3.0)
c = a + b  # __add__ method is called and new value created with 
           # (self, other) being the _children argument
d = a*b + c

print(c)
print(c._prev)
print(d._prev) # but only gives most recent 