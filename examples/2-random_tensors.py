import cgrad

print("\nGenrate the random tensors\n")
a = cgrad.rand((3,2,2))
print(a)

print("\nGenerate the zeros tensors\n")
a = cgrad.zeros((1,2,3))
print(a)

print("\nGenerate the ones tensors\n")
a = cgrad.ones((2,3,4))
print(a)