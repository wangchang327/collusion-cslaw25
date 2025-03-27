N = 100  # number of repeats
T_range = (
    [x * 100 for x in [2, 4, 6, 8, 10]]
    + [x * 1000 for x in [2, 4, 6, 8, 10]]
    + [x * 10000 for x in [2, 4, 6, 8, 10]]
    + [x * 100000 for x in range(2, 11)]
)


# compute the probability of winning
# my: the seller's price
# opp: the opponent's price
def cdf(my, opp):
    x = my - opp
    y = my
    if x >= 0:
        if y <= x:
            the = 0.5 * (1 - x) * (1 - x)
        else:
            the = 0.5 * (y - 2 * x + 1) * (1 - y)
    else:
        the = 0.5 * (y - x + 1) * (1 + x - y) - x

    return the
