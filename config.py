class Config:
    def __init__(self, shape):

        self.dim = {
            'x': shape[1],
            'y': shape[2]
        }
        self.nFrames = shape[0]
