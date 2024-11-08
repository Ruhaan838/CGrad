import cgrad
from cgrad import AutoGrad

a = cgrad.Tensor([1,2,3], requires_grad=True)
b = cgrad.Tensor([1,2,3], requires_grad=True)

with AutoGrad.no_grad():
    print("\nInside the no_grad")
    d = a * b
    e = d * 10
    f = e * 10
    print(f"d:, requires_grad:{d.requires_grad}, grad:{d.grad}")
    print(f"e:, requires_grad:{e.requires_grad}, grad:{e.grad}")
    print(f"f:, requires_grad:{f.requires_grad}, grad:{f.grad}")


print("\noutsize the no_grad")
c = a + b
c.sum().backward()
print(f"c:, requires_grad:{c.requires_grad}, grad:{c.grad}\n")
