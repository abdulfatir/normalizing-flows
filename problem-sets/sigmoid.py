class Sigmoid(tfb.Bijector):
    def __init__(self, validate_args=False, name="sigmoid"):
        super(Sigmoid, self).__init__(
            forward_min_event_ndims=0,
            validate_args=validate_args,
            name=name)

    def _forward(self, x):
        return 1./(1 + tf.exp(-x))

    def _inverse(self, y):
        return -tf.log(1/y - 1)

    def _inverse_log_det_jacobian(self, y):
        I = tf.ones_like(y)
        J_inv = 1/(y - y ** 2) * I
        log_abs_det_J_inv = tf.log(tf.abs(J_inv))
        return log_abs_det_J_inv