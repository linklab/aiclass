
print('직각삼각형 그리기')
d = float(input('변의 길이 : '))
for i in range(int(d) + 1):
    print('* ' * i)
area = float((d ** 2) / 2)
print('넓이 : %s' % area)

