def fun():
    global g
    global sum
    for i in range(0, 10):
        print(i + g)
    sum = g + 999


g = 10
fun()
print(sum)
