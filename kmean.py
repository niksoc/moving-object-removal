import sampler, pickle
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.cluster import KMeans
import numpy as np
from config import Config

frames = None
configs = None

def pixelDensity(x, y):
    vec = frames[:,x, y]
    binwidth = 25

    plt.hist(vec, binwidth, normed=True)
    plt.xlim((min(vec), max(vec)))

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

def getBg(clip, cached=None):
    bg = []

    frames = sampler.getFrames(clip)
    configs = Config(frames.shape)
    print "Frame Dimension: ", frames.shape

    if cached == None:
        for i in xrange(configs.dim['x']):
            for j in xrange(configs.dim['y']):
                print "Pixel: %i, %i"%(i,j)
                vec = frames[:, i, j]

                cent, labels = cluster(vec)
                # probs = probability(vec, cent)
                bg.append(getBgPixel(vec, labels, cent[0][0], cent[1][0]))

        bg = np.array(bg)
        bg = bg.reshape(configs.dim['x'], configs.dim['y'])

        with open('bin/bg_kmeans_%s.pkl'%clipName.split('/')[-1], 'wb') as f:
            pickle.dump(bg, f)
        return bg
        # plt.imshow(bg, cmap='gray')
    else:
        print "!!! Extracting BG from cached files"
        with open(cached, 'rb') as f:
            bg = pickle.load(f)
        return bg

if __name__ == '__main__':
    getBg("data/nikwalking.mp4", "bin/k_means_nikwalking.pkl")

    # X, Y = 225, 300
    #
    # vec = frames[:,X,Y]
    # pixelDensity(X, Y)
    # print cluster(vec)

    plt.show()
