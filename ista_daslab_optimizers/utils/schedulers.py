import torch

def ema_standard_schedule(m, g, beta):
    """
    Implements the standard EMA: m_new = beta * m_old + (1 - beta) * g
    :param m: momentum buffer
    :param g: gradient
    :param beta: EMA coefficient
    """
    m.lerp_(g, 1 - beta)

def ema_delayed_decay_schedule(m, g, beta, beta_prev, t, T_decay, alpha):
    """
    This version is proposed by Mher Safaryan in June 2025 while a postdoc @ ISTA:

    beta_0 = 1 (tracks largest weight)
    alpha >= 0 (sub-schedule slope)

    if t == 1 or t % T_decay == 0:
        m_t = beta * m_t-1 + (1 - beta) * g
        beta_t = 1 - beta
    else:
        m_t = (1 / (1 + alpha + beta_t-1)) * m_t-1 + (alpha + beta_t-1) / (1 + alpha + beta_t-1) * g
        beta_t = (alpha + beta_t-1) / (1 + alpha + beta_t-1)

    :param m: momentum buffer
    :param g: gradient buffer
    :param beta: EMA coefficient
    :param beta_prev: previous EMA coefficient
    :param alpha: slope (use values between 0.001 and 0.007)
    :param T_decay: decay interval
    :return: returns beta_t
    """

    if t == 1 or t % T_decay == 0:
        ema_standard_schedule(m, g, beta)
        return 1 - beta
    else:
        beta_t = (alpha + beta_prev) / (1 + alpha + beta_prev)
        ema_standard_schedule(m, g, 1-beta_t)
        return beta_t
