N = 10  # number of repeats
T = 100000  # number of rounds


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
