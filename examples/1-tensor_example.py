import cgrad

print("create our own custom tensor")
a = cgrad.Tensor([[1,2,3],[1,2,3]])
print(f"\nTensor: {a}")
print(f"Shape of Tensor:{a.shape}")
print(f"Dim of Tensor:{a.ndim}\n")

print("create tensor using int or float")
g = cgrad.Tensor(1)
print(f"\nTensor: {g}")
print(f"Shape of Tensor:{g.shape}")
print(f"Dim of Tensor:{g.ndim}\n")

print("add two tensor element wise addition")
b = cgrad.Tensor([4,5,6])
c = a + b
print(f"\nresult Tensor:{c}")
print(f"Shape of result Tensor:{c.shape}")
print(f"Dim of result Tensor:{c.ndim}\n")
print("add using .add")
c = cgrad.add(a,b)
print(f"\nresult Tensor:{c}")
print(f"Shape of result Tensor:{c.shape}")
print(f"Dim of result Tensor:{c.ndim}\n")

print("broadcasting rule for addition")
d = cgrad.Tensor([[1,2,3],[3,4,5]])
e = d + 100
print(f"\nResult Tensor:{e}")
print(f"Shape of result Tensor:{e.shape}")
print(f"Dim of Result Tensor:{e.ndim}\n")

print("multiplication ele-wise")
c = a * b
print(f"\nResult Tensor:{c}")
print(f"Shape of result Tensor:{c.shape}")
print(f"Dim of Result Tensor:{c.ndim}\n")

print("broadcasting rule for mutiply")
d = cgrad.Tensor([[1,2,3],[3,4,5]])
e = d * 5
print(f"\nResult Tensor:{e}")
print(f"Shape of result Tensor:{e.shape}")
print(f"Dim of Result Tensor:{e.ndim}\n")

print("subtract two tensor")
f = a - cgrad.Tensor([1,2,3])
print(f"\nResult Tensor{f}")
print(f"Shape of result Tensor:{f.shape}")
print(f"Dim of Result Tensor:{f.ndim}\n")

print("broadcasting rule for subtraction")
f = a - 10
print(f"\nResult Tensor{f}")
print(f"Shape of result Tensor:{f.shape}")
print(f"Dim of Result Tensor:{f.ndim}\n")

print("division of two tensor")
t1 = cgrad.Tensor([[1,2,3], [2,3,4]])
t2 = cgrad.Tensor([[2,3,4], [3,4,4]])
r = t1 / t2
print(f"\nResult Tensor{r}")
print(f"Shape of result Tensor:{r.shape}")
print(f"Dim of Result Tensor:{r.ndim}\n")

print("broadcasting rule for division")
t1 = cgrad.Tensor([100, 200, 300])
r = t1 / 100
print(f"\nResult Tensor{r}")
print(f"Shape of result Tensor:{r.shape}")
print(f"Dim of Result Tensor:{r.ndim}\n")