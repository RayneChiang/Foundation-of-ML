from matplotlib import pyplot as plt
from shapely.geometry.polygon import LinearRing, Polygon
import numpy as np
if __name__ == '__main__':
    poly = Polygon([(0, 1), (0.35, 0.55), (0.7, 0.1), (0.15, 0.95), (0.5, 0.5),  (0.85, 0.05),
                    (0.3, 0.9), (0.65, 0.45), (1, 0)])
    x, y = poly.exterior.xy
    # result_utility = np.add(x,y)
    # result_nbs = np.multiply(x,y)
    # print(np.argmax(result_utility))
    # print(np.argmax(result_nbs))
    plt.plot(x, y)

    plt.savefig("negotiation.png")