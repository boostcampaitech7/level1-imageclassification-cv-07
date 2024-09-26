import pdb

def add(a, b):
    return a + b

def multiply(x, y):
    result = add(x, y)
    pdb.set_trace()  # 중단점 설정
    return result * 2

result = multiply(5, 3)
print(f"Final result: {result}")

