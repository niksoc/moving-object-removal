import sampler
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.cluster import KMeans
import numpy as np
from config import Config

frames = sampler.getFrames('data/nikwalking.mp4')
configs = Config(frames.shape)
print "Frame Dimension: ", frames.shape

def pixelDensity(x, y):
    vec = frames[:,x, y]
    binwidth = 1

    plt.hist(vec, bins=range(min(vec), max(vec) + binwidth, binwidth), normed=True)
    plt.xlim((min(vec), max(vec)))

    # mean = np.mean(vec)
    # variance = np.var(vec)
    # sigma = np.sqrt(variance)
    # x = np.linspace(min(vec), max(vec), 100)
    # plt.plot(x, mlab.normpdf(x, mean, sigma))

def dist(x, y):
    return abs(x-y)

def probability(vec, centers):
    probability = []
    c1, c2 = centers[0][0], centers[1][0]
    for val in vec:
        p1 = 1- dist(val, c1)/(dist(val, c2) + dist(val, c1))
        p2 = 1- dist(val, c2)/(dist(val, c2) + dist(val, c1))

        probability.append(p1 if p1>p2 else p2)
    return probability


def cluster(x):
    x = x.reshape(configs.nFrames, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(x)

    nC1 = kmeans.label_.shape[0] - np.sum(kmeans.label_.shape[0])
    nC0 = kmeans.label_.shape[0] - nC1

    if nC1 > nC2:
        print "Centroid 1: Background\nCentroid 0: Foreground"
        return np.flip(kmeans.cluster_centers_)
    else:
        print "Centroid 0: Background\nCentroid 1: Foreground"
        return kmeans.cluster_centers_

def getBackgroundPixel(vec, bCent, fCent):
    return bCent

if __name__ == '__main__':
    X, Y = 200, 300
    vec = frames[:, X, Y]
    # pixelDensity(200, 200)

    cent = cluster(vec)
    probs = probability(vec, cent)

    plt.show()
