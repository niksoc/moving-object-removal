import sampler, pickle
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
    binwidth = 25

    plt.hist(vec, binwidth, normed=True)
    plt.xlim((min(vec), max(vec)))

    # mean = np.mean(vec)
    # variance = np.var(vec)
    # sigma = np.sqrt(variance)
    # x = np.linspace(min(vec), max(vec), 100)
    # plt.plot(x, mlab.normpdf(x, mean, sigma))

# def dist(x, y):
#     return abs(x-y)
#
# def probability(vec, centers):
#     probability = []
#     c1, c2 = centers[0][0], centers[1][0]
#     for val in vec:
#         p1 = 1- dist(val, c1)/(dist(val, c2) + dist(val, c1))
#         p2 = 1- dist(val, c2)/(dist(val, c2) + dist(val, c1))
#
#         probability.append(p1 if p1>p2 else p2)
#     return probability


def cluster(x):
    x = x.reshape(configs.nFrames, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(x)

    nC0 = kmeans.labels_.shape[0] - np.sum(kmeans.labels_)
    nC1 = kmeans.labels_.shape[0] - nC0
    print "C(Cluster 0):", nC0
    print "C(Cluster 1):", nC1

    if nC1 > nC0:
        print "Centroid 1: Background\nCentroid 0: Foreground"
        return kmeans.cluster_centers_[::-1], kmeans.labels_
    else:
        print "Centroid 0: Background\nCentroid 1: Foreground"
        return kmeans.cluster_centers_, kmeans.labels_

def getBgPixel(vec, labels, bCent, fCent):
    return bCent

def getBg():
    bg = []
    for i in xrange(configs.dim['x']):
        for j in xrange(configs.dim['y']):
            print "Pixel: %i, %i"%(i,j)
            vec = frames[:, i, j]

            cent, labels = cluster(vec)
            # probs = probability(vec, cent)
            bg.append(getBgPixel(vec, labels, cent[0][0], cent[1][0]))

    bg = np.array(bg)
    bg = bg.reshape(configs.dim['x'], configs.dim['y'])

    with open('bin/bg_kmeans_hand.pkl', 'wb') as f:
        pickle.dump(bg, f)
    plt.imshow(bg, cmap='gray')

if __name__ == '__main__':
    getBg()
    # X, Y = 225, 300
    #
    # vec = frames[:,X,Y]
    # pixelDensity(X, Y)
    # print cluster(vec)

    plt.show()
