# a, b integers
# what is the most frequent sum of integers for number between a and b


def sum_digits(i: int):
    return sum(map(int, iter(str(i))))


def solve(a: int, b: int):
    if a == b:
        return {sum_digits(i=a)}
    if b < a:
        return solve(a=b, b=a)
    sa, sb = str(a), str(b)
    if len(sa) == len(sb) and sa[0] == sb[0]:
        short_a, short_b = int(sa[1:]), int(sb[1:])
        s = solve(a=short_a, b=short_b)
        i = int(sa[0])
        return {i + ss for ss in s}
    return {len(str(a)), len(str(b))}


if __name__ == '__main__':
    def show_solve(a: int, b: int):
        print(f"{a}\t{b}:\t{solve(a=a, b=b)}")

    show_solve(a=1, b=100)
    show_solve(a=1, b=10000)
    show_solve(a=1, b=1000000)
    show_solve(a=5000, b=5004)
