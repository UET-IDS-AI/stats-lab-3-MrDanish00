import numpy as np
import math


# =========================================================
# QUESTION 1 – Card Experiment
# =========================================================

def card_experiment():
    # Events:
    # A = first card is Ace
    # B = second card is Ace (without replacement)

    P_A = 4 / 52
    P_B = 4 / 52
    P_B_given_A = 3 / 51
    P_AB = P_A * P_B_given_A

    # Simulation
    np.random.seed(42)
    trials = 200000

    count_A = 0
    count_B_given_A = 0

    for _ in range(trials):
        deck = np.zeros(52)
        deck[:4] = 1  # 1 = Ace, 0 = not Ace
        np.random.shuffle(deck)

        first = deck[0]
        second = deck[1]

        if first == 1:
            count_A += 1
            if second == 1:
                count_B_given_A += 1

    empirical_P_A = count_A / trials
    empirical_P_B_given_A = count_B_given_A / count_A

    absolute_error = abs(P_B_given_A - empirical_P_B_given_A)

    return (
        P_A,
        P_B,
        P_B_given_A,
        P_AB,
        empirical_P_A,
        empirical_P_B_given_A,
        absolute_error
    )


# =========================================================
# QUESTION 2 – Bernoulli
# =========================================================

def bernoulli_lightbulb(p=0.05):
    theoretical_P_X_1 = p
    theoretical_P_X_0 = 1 - p

    np.random.seed(42)
    samples = np.random.binomial(1, p, 100000)

    empirical_P_X_1 = np.mean(samples)

    absolute_error = abs(theoretical_P_X_1 - empirical_P_X_1)

    return (
        theoretical_P_X_1,
        theoretical_P_X_0,
        empirical_P_X_1,
        absolute_error
    )


# =========================================================
# QUESTION 3 – Binomial
# =========================================================

def binomial_bulbs(n=10, p=0.05):
    def binom_pmf(k):
        return math.comb(n, k) * (p**k) * ((1-p)**(n-k))

    theoretical_P_0 = binom_pmf(0)
    theoretical_P_2 = binom_pmf(2)
    theoretical_P_ge_1 = 1 - theoretical_P_0

    np.random.seed(42)
    samples = np.random.binomial(n, p, 100000)

    empirical_P_ge_1 = np.mean(samples >= 1)

    absolute_error = abs(theoretical_P_ge_1 - empirical_P_ge_1)

    return (
        theoretical_P_0,
        theoretical_P_2,
        theoretical_P_ge_1,
        empirical_P_ge_1,
        absolute_error
    )


# =========================================================
# QUESTION 4 – Geometric
# =========================================================

def geometric_die():
    p = 1 / 6

    theoretical_P_1 = p
    theoretical_P_3 = ((5/6)**2) * p
    theoretical_P_gt_4 = (5/6)**4

    np.random.seed(42)
    samples = np.random.geometric(p, 200000)

    empirical_P_gt_4 = np.mean(samples > 4)

    absolute_error = abs(theoretical_P_gt_4 - empirical_P_gt_4)

    return (
        theoretical_P_1,
        theoretical_P_3,
        theoretical_P_gt_4,
        empirical_P_gt_4,
        absolute_error
    )


# =========================================================
# QUESTION 5 – Poisson
# =========================================================

def poisson_customers(lam=12):
    def poisson_pmf(k):
        return math.exp(-lam) * (lam**k) / math.factorial(k)

    theoretical_P_0 = poisson_pmf(0)
    theoretical_P_15 = poisson_pmf(15)

    theoretical_P_ge_18 = 1 - sum(poisson_pmf(k) for k in range(18))

    np.random.seed(42)
    samples = np.random.poisson(lam, 100000)

    empirical_P_ge_18 = np.mean(samples >= 18)

    absolute_error = abs(theoretical_P_ge_18 - empirical_P_ge_18)

    return (
        theoretical_P_0,
        theoretical_P_15,
        theoretical_P_ge_18,
        empirical_P_ge_18,
        absolute_error
    )
