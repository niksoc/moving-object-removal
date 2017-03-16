import kmean, sampler, math
from config import Config

frames, configs = None, None

def test(target, bg):
    err = 0
    for i in xrange(configs.dim['x']):
        for j in xrange(configs.dim['y']):
            err += (target[i, j]-bg[i, j])**2
    return err/(configs.dim['x']*configs.dim['y'])**0.5

if __name__ == '__main__':
    clip = "data/nikwalking.mp4"
    cached = "bin/bg_kmeans_nikwalking.pkl"
    targetFrame = 0

    frames = sampler.getFrames(clip)
    configs = Config(frames.shape)

    bg = kmean.getBg(clip, cached)
    target = frames[targetFrame,:,:]

    print "RMS Error:", test(target, bg)
