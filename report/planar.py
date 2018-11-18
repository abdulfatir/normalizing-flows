m = lambda x: -1 + tf.log(1 + tf.exp(x))
h = lambda x: tf.tanh(x)
h_prime = lambda x: 1 - tf.tanh(x) ** 2
base_dist = tfd.MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 1.])
z_0 = base_dist.sample(500)
z_prev = z_0
sum_log_det_jacob = 0.
for i in range(K):
    with tf.variable_scope('layer_%d' % i):
        u = tf.get_variable('u', dtype=tf.float32, shape=(1, 2))
        w = tf.get_variable('w', dtype=tf.float32, shape=(1, 2))
        b = tf.get_variable('b', dtype=tf.float32, shape=())
        u_hat = (m(tf.tensordot(w, u, 2)) - tf.tensordot(w, u, 2)) * \
            (w / tf.norm(w)) + u
        affine = h_prime(tf.expand_dims(
            tf.reduce_sum(z_prev * w, -1), -1) + b) * w
        sum_log_det_jacob += tf.log(tf.abs(1 +
                                           tf.reduce_sum(affine * u_hat, -1)))
        z_prev = z_prev + u_hat * \
            h(tf.expand_dims(tf.reduce_sum(z_prev * w, -1), -1) + b)
z_k = z_prev
log_q_k = base_dist.log_prob(z_0) - sum_log_det_jacob
log_p = tf.log(true_density(z_k))

kl = tf.reduce_mean(log_q_k - log_p, -1)
