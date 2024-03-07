from .Envelope import Envelope
import numpy as np


class BesselBasisLayer:
    def __init__(self, num_radial, cutoff, envelope_exponent=5):
        self.num_radial = num_radial  # num_radial=6
        self.inv_cutoff = np.array(1 / cutoff, dtype=np.float32)  # cutoff=5   从一个类张量的物体中创建一个常数张量
        self.envelope = Envelope(envelope_exponent)

        # Initialize frequencies at canonical positions 在规范位置初始化频率
        def freq_init(shape, dtype):
            return np.array(np.pi * np.arange(1, shape + 1, dtype=np.float32), dtype=dtype)
        self.frequencies = self.add_weight(name="frequencies", shape=self.num_radial,
                                           dtype=np.float32, initializer=freq_init, trainable=True)

    def call(self, inputs):
        d_scaled = inputs * self.inv_cutoff

        # Necessary for proper broadcasting behaviour  这是正确广播行为的必要条件
        d_scaled = np.expand_dims(d_scaled, -1)

        d_cutoff = self.envelope(d_scaled)
        return d_cutoff * np.sin(self.frequencies * d_scaled)