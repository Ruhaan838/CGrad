import cgrad as cg

#create our own custom tensor
a = cg.Tensor([1,2,3])
print(f"\nTensor: {a}")
print(f"Shape of Tensor:{a.shape}")
print(f"Dim of Tensor:{a.ndim}\n")

#add two tensor element wise addition
b = cg.Tensor([4,5,6])
c = a + b
print(f"\nresult Tensor:{c}")
print(f"Shape of result Tensor:{c.shape}")
print(f"Dim of result Tensor:{c.ndim}\n")

# broadcast rule for addition
d = cg.Tensor([[1,2,3],[3,4,5]])
e = d + 100
print(f"\nResult Tensor:{e}")
print(f"Shape of result Tensor:{e.shape}")
print(f"Dim of Result Tensor:{e.ndim}\n")

#multiplication ele-wise
c = a * b
print(f"\nResult Tensor:{c}")
print(f"Shape of result Tensor:{c.shape}")
print(f"Dim of Result Tensor:{c.ndim}\n")

# broadcast rule for mutiply
d = cg.Tensor([[1,2,3],[3,4,5]])
e = d * 5
print(f"\nResult Tensor:{e}")
print(f"Shape of result Tensor:{e.shape}")
print(f"Dim of Result Tensor:{e.ndim}\n")
