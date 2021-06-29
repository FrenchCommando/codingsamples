def generate_tribonacci_numbers():
    a, b, c = 0, 0, 1
    while True:
        a, b = b, c = c, a + b + c
        # Yield an infinite stream of Tribonacci numbers! The next value of the sequence will be c + b + a.
        yield a


def is_tribonacci(num):
    if num == 0 or num == 1:
        return True
    else:
        it = generate_tribonacci_numbers()
        a = next(it)
        while a < num:
            a = next(it)
        return a == num


if __name__ == '__main__':
    for i in range(10):
        print(f"{i}\t{is_tribonacci(num=i)}")
