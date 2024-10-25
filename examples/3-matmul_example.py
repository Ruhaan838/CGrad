import cgrad

print("\nMultiply the 2D mat\n")

a = cgrad.randn((2,2)) # seed = 42
b = cgrad.randn((2,2), seed = 32) # seed = 32

c = a @ b

print(c)

print("\nMultiply the 3D mat\n")

a = cgrad.randn((3,3)) # seed = 42
b = cgrad.randn((3,3), seed = 76) # seed = 76

c = a @ b

print(c)

print("\nMultiply the ND mat\n")

a = cgrad.randn((1,2,3)) # seed = 42
b = cgrad.randn((4,3,2), seed = 65) # seed = 65

c = a @ b

print(c)



