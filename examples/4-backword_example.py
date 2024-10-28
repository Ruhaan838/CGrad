import cgrad

x1 = cgrad.Tensor([1,3])
x2 = cgrad.Tensor([1,2], require_grad=True)

w1 = cgrad.Tensor([3,5])
w2 = cgrad.Tensor([6,7], require_grad=True)

b = cgrad.Tensor([8,10], require_grad=True)

x1w1 = x1 * w1
x2w2 = x2 * w2
x1w1x2w2 = x1w1 + x2w2
n = x1w1x2w2 + b

print("-"*10, "forward pass", "-"*10)
print("x1:",x1)
print("w1:",w1)
print("x2:",x2)
print("w2:",w2)
print("b:",b)
print("x1w1:",x1w1)
print("x2w2:",x2w2)
print("x1w1x2w2:",x1w1x2w2)
print("n:",n)

print("-"*10 ,"backward pass", "-"*10)
n.backward()
print("x1w1x2w2:",x1w1x2w2.grad)
print("b:",b.grad)
print("x1w1:",x1w1.grad)
print("x2w2",x2w2.grad)
print("x1:",x1.grad)
print("x2:",x2.grad)
print("w1:",w1.grad)
print("w2:",w2.grad)