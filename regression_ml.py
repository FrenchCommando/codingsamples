import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVR
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def linear(d_quotes):

    """{k: {"Ask": 0.1, "Bid": 0.05"}}"""
    strikes = []
    quotes = []
    labels = []
    for strike, quote in d_quotes.items():
        if "Ask" in quote:
            strikes.append(strike)
            quotes.append(quote["Ask"])
            labels.append(1)
        if "Bid" in quote:
            strikes.append(strike)
            quotes.append(quote["Bid"])
            labels.append(0)
    X = np.array([strikes, quotes]).T
    y = np.array(labels)
    # print(X.shape)
    # print(y.shape)
    # print(X)
    print(y)
    clf = make_pipeline(
        StandardScaler(), RidgeClassifier(max_iter=1000, tol=1e-10),
        # StandardScaler(), SVR(C=1, epsilon=0.01, kernel='linear'),
    )
    clf.fit(X, y)
    print(clf.predict(X))
    # print(s.intercept_)
    # print(s.coef_)
    print(clf.get_params())
    return clf


def display_stuff(d, out):
    matplotlib.use('TkAgg')  # Or 'Qt5Agg' or 'Qt4Agg'
    strikes = np.array(list(d.keys()))
    asks = np.array(list(u["Ask"] for u in d.values()))
    bids = np.array(list(u["Bid"] for u in d.values()))

    def find_separation(k):
        q_max = d[k]["Ask"] + 10
        q_min = d[k]["Bid"] - 10
        q_range = np.linspace(q_min, q_max, 100)
        p = [[k, u] for u in q_range]
        pp = out.predict(p)
        # print(p)
        # print(pp)
        q_min = max((q for u, q in zip(pp, q_range) if u == 0), default=np.nan)
        q_max = min((q for u, q in zip(pp, q_range) if u == 1), default=np.nan)
        return 0.5 * (q_min + q_max)

    separation = np.array(list(find_separation(k=u) for u in strikes))

    plt.figure("Quotes")
    plt.scatter(strikes, asks, label="Ask", marker=7)
    plt.scatter(strikes, bids, label="Bid", marker=6)
    plt.scatter(strikes, separation, label="Separation", marker='x')
    plt.legend()
    plt.ylabel("Quote")
    plt.xlabel("Strike")
    plt.minorticks_on()
    plt.grid(visible=True, alpha=0.8, which='major')
    plt.grid(visible=True, alpha=0.2, which='minor')
    plt.title("Quotes")

    plt.savefig("Quotes.png")

    plt.figure("Diffs")
    plt.scatter(strikes, asks-separation, label="Ask", marker=7)
    plt.scatter(strikes, bids-separation, label="Bid", marker=6)
    plt.legend()
    plt.ylabel("Diffs")
    plt.xlabel("Strike")
    plt.minorticks_on()
    plt.grid(visible=True, alpha=0.8, which='major')
    plt.grid(visible=True, alpha=0.2, which='minor')
    plt.title("Diffs")

    plt.savefig("Diffs.png")

    plt.show()


def bid_ask_example():

    d = {
        5560.0: {'Ask': np.float64(178.89999999999998), 'Bid': np.float64(174.6)},
        5570.0: {'Ask': np.float64(169.10000000000002), 'Bid': np.float64(164.70000000000002)},
        5575.0: {'Ask': np.float64(164.10000000000002), 'Bid': np.float64(159.8)},
        5580.0: {'Ask': np.float64(159.3), 'Bid': np.float64(154.89999999999998)},
        5590.0: {'Ask': np.float64(149.49999999999997), 'Bid': np.float64(145.10000000000002)},
        5600.0: {'Ask': np.float64(139.3), 'Bid': np.float64(135.60000000000002)},
        5610.0: {'Ask': np.float64(129.89999999999998), 'Bid': np.float64(125.50000000000003)},
        5625.0: {'Ask': np.float64(114.69999999999999), 'Bid': np.float64(110.50000000000003)},
        5650.0: {'Ask': np.float64(90.19999999999999), 'Bid': np.float64(86.40000000000003)},
        5675.0: {'Ask': np.float64(65.60000000000002), 'Bid': np.float64(61.39999999999998)},
        5700.0: {'Ask': np.float64(41.30000000000001), 'Bid': np.float64(37.80000000000001)},
        5725.0: {'Ask': np.float64(16.80000000000001), 'Bid': np.float64(12.399999999999977)},
        5750.0: {'Ask': np.float64(-7.800000000000011), 'Bid': np.float64(-11.699999999999989)},
        5775.0: {'Ask': np.float64(-32.10000000000002), 'Bid': np.float64(-36.60000000000002)},
        5800.0: {'Ask': np.float64(-56.799999999999955), 'Bid': np.float64(-60.80000000000001)},
        5825.0: {'Ask': np.float64(-81.30000000000001), 'Bid': np.float64(-85.70000000000002)},
    }
    out = linear(d_quotes=d)
    display_stuff(d, out)


if __name__ == '__main__':
    bid_ask_example()
