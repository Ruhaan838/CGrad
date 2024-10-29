import cgrad

print("\nMultiply the 2D mat\n")

a = cgrad.randn((2,2)) 
b = cgrad.randn((2,2)) 

c = a @ b

print(c)

print("\nMultiply the 3D mat\n")

a = cgrad.randn((3,3)) 
b = cgrad.randn((3,3)) 

c = a @ b

print(c)

print("\nMultiply the ND mat\n")

a = cgrad.randn((1,2,3)) 
b = cgrad.randn((4,3,2)) 

c = a @ b

print(c)



