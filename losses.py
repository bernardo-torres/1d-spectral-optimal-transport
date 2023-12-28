import torch
from utils import safe_divide, safe_log
from features import compute_mag, get_transform
import functools


def mean_difference(target, value, loss_type="L1", weights=None, dims=None):
    """Common loss functions.
    Args:
      target: Target tensor.
      value: Value tensor.
      loss_type: One of 'L1', 'L2', or 'COSINE'.
      weights: A weighting mask for the per-element differences.
    Returns:
      The average loss.
    Raises:
      ValueError: If loss_type is not an allowed value.
    """
    difference = target - value
    weights = 1.0 if weights is None else weights
    loss_type = loss_type.upper()
    if loss_type == "L1":
        if dims is None:
            dims = [x for x in range(len(difference.shape))]
        return torch.mean(torch.abs(difference * weights), dim=dims)
        # return torch.abs(difference * weights)

    elif loss_type == "L2":
        if dims is None:
            dims = [x for x in range(len(difference.shape))]
        return torch.mean(difference**2 * weights, dim=dims)
        # return difference**2 * weights
    # elif loss_type == 'COSINE':
    # return tf.losses.cosine_distance(target, value, weights=weights, axis=-1)
    else:
        raise ValueError("Loss type ({}), must be " '"L1", "L2" '.format(loss_type))


class MeanDifference(torch.nn.Module):
    def __init__(self, loss_type="L1"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, x, y, weights=None, sort=False, **kwargs):
        if sort:
            x, _ = torch.sort(x, dim=-1)
            y, _ = torch.sort(y, dim=-1)
        return mean_difference(
            x,
            y,
            loss_type=self.loss_type,
            weights=weights,
            dims=kwargs.get("dims", None),
        )


class KL(torch.nn.Module):
    def __init__(self, eps=1e-10, **kwargs):
        super().__init__()
        self.eps = eps
        self.reverse = kwargs.get("reverse", False)

    # KL(input, target)
    # true, estimated
    def forward(self, input, target, **kwargs):
        original_shape = input.shape[:-1]
        if input.ndim == 3:
            input = input.reshape(-1, input.shape[-1])
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[-1])

        if self.reverse:
            input, target = target, input

        # Normalize input and target over each feature
        input = safe_divide(input, torch.sum(input, dim=-1, keepdim=True))
        target = safe_divide(target, torch.sum(target, dim=-1, keepdim=True))

        logx = torch.log(input + self.eps)
        logy = torch.log(target + self.eps)

        kl = input * (logx - logy)
        kl = torch.sum(kl, dim=-1)
        # Reshape to batch, time
        kl = kl.reshape(original_shape)
        return torch.mean(kl, dim=kwargs.get("dims", None))


class Wasserstein1D(torch.nn.Module):
    def __init__(
        self,
        p=1,
        fixed_x=None,
        require_sort=True,
        log_scaled_x=False,
        **kwargs,
    ):
        """ Loss class to compute the 1D Wasserstein distance between two distributions. It handles weight normalization and
            batch processing before computing the W distance.
            
            Args:
                p: Order of the Wasserstein distance.
                fixed_x: Number of points to use for the quantile function if known in advance to be fixed. 
                    If None, x_pos and y_pos must be provided on forward.
                require_sort: If True, sort u_values and v_values before computing the loss.
                log_scaled_x: If True, scale x_pos and y_pos logarithmically.
                square_dist: If True, square the weight values before computing the loss.
                hinge: If set, use a hinge loss to only penalize values above a certain threshold.

                kwargs: Additional arguments to pass to wasserstein_1d.

            """
        super().__init__()

        self.p = p
        self.require_sort = require_sort
        self.log_scaled_x = log_scaled_x  # TODO implement log scaling inside, currently done outside the forward call
        self.dont_normalize = kwargs.get("dont_normalize", False)
        self.limit_quantile_range = kwargs.get("limit_quantile_range", False)
        self.hinge = kwargs.get("hinge", False)
        self.square_dist = kwargs.get("square_dist", False)

        # If fixed_x is an integer, register buffer with evenly spaces values from 0, 1
        if fixed_x is not None:
            self.register_buffer("fixed_x", torch.linspace(0, 1, fixed_x))
        else:
            self.register_buffer("fixed_x", None)

    def forward(self, x, y, x_pos=None, y_pos=None, **kwargs):
        """ Compute the 1D Wasserstein distance between two (batched) distributions.
            Args:
                x: Tensor of shape (batch, time, features) or (batch, features).
                y: Tensor of shape (batch, time, features) or (batch, features).
                x_pos: Tensor of shape (batch, time, features) or (batch, features) containing the positions of the weights of the first distribution.
                    if None, use self.fixed_x.
                y_pos: Tensor of shape (batch, time, features) or (batch, features) containing the positions of the weights of the second distribution.
                    if None, use self.fixed_x.
                kwargs: Additional arguments to pass to wasserstein_1d. Options are:
                    dont_normalize: If True, don't normalize the second distribution to sum to 1 but scale wrt the first distribution which is normalized.
                    return_quantiles: If True, return the quantiles of u and v.
                    limit_quantile_range: If True, set the distance to 0 if the quantile is greater than 1 (which can happen if dont_normalize is True).

                    In the paper, dont_normalize and limit_quantile_range refer to the "frequency cutoff".

            Returns:
                The 1D Wasserstein distance between the two distributions.
                
                """
        

        if (x_pos is None or y_pos is None) and self.fixed_x is None:
            raise ValueError("If fixed_x is not provided, x_pos and y_pos must be provided")

        x_pos_ = self.fixed_x if x_pos is None else x_pos
        y_pos_ = self.fixed_x if y_pos is None else y_pos

        original_shape = x.shape[:-1]
        if x.ndim == 3:  # batch, time, features
            x = x.reshape(-1, x.shape[-1])
        if y.ndim == 3:  # batch, time, features
            y = y.reshape(-1, y.shape[-1])  # batch * time, features
        if x_pos_.ndim == 3:
            x_pos_ = x_pos_.reshape(-1, x_pos_.shape[-1])
        if y_pos_.ndim == 3:
            y_pos_ = y_pos_.reshape(-1, y_pos_.shape[-1])

        if x_pos_.ndim == 1:
            x_pos_ = x_pos_.unsqueeze(0).expand_as(x)
        if y_pos_.ndim == 1:
            y_pos_ = y_pos_.unsqueeze(0).expand_as(y)

        if self.square_dist:
            x = x**2
            y = y**2

        # Normalize x
        total_mass_x = torch.sum(x, dim=1, keepdim=True)
        x = safe_divide(x, total_mass_x)

        if kwargs.get("dont_normalize", False) or self.dont_normalize:
            # Don't normalize y but scale wrt x which is normalized
            y = safe_divide(y, total_mass_x)
        else:
            y = safe_divide(y, torch.sum(y, dim=1, keepdim=True))

        loss = wasserstein_1d(
            x_pos_,
            y_pos_,
            u_weights=x,
            v_weights=y,
            p=self.p,
            require_sort=self.require_sort,
            return_quantiles=kwargs.get("return_quantiles", False),
            limit_quantile_range=kwargs.get("limit_quantile_range", False)
            or self.limit_quantile_range,
        )

        if kwargs.get("return_quantiles", False):
            # Reshape every item in loss to batch, time, features
            loss = [l.reshape(original_shape + (-1,)) for l in loss]
            return loss

        if self.hinge:
            # hinge loss to only use high values
            loss = torch.nn.functional.relu(loss - kwargs.get("hinge", 0.0))

        # Reshape to batch, time
        loss = loss.reshape(original_shape)

        # Average over time frames
        return torch.mean(loss, dim=kwargs.get("dims", None))


def quantile_function(qs, cws, xs):
    n = xs.shape[1]
    # idx = torch.searchsorted(cws, qs).transpose(-2,-1)
    # cws = cws.contiguous()
    # qs = qs.contiguous()
    idx = torch.searchsorted(cws, qs)
    return torch.take_along_dim(xs, torch.clamp(idx, 0, n - 1), dim=1)


def wasserstein_1d(
    u_values,
    v_values,
    u_weights=None,
    v_weights=None,
    p=1,
    require_sort=True,
    return_quantiles=False,
    limit_quantile_range=False,
):
    """ Approximates the 1D Wasserstein distance between two distributions by a sum of distances between quantiles.
        We assume  (u_weights, v_weights)  belong to the space of probability vectors, $i.e.$ $u_weights \in \Sigma_n$ and 
        $v_weights \in \Sigma_m$, for $\Sigma_n = \left\{\mathbf{a} \in \mathbb{R}^n_+ ; \sum_{i=1}^n \mathbf{a}_i = 1 \right\}$. 
        That means the weights are normalized to sum to 1 and are non-negative.

        The Wasserstein distance between two one dimensional distributions can be expressed in closed form as [1, prop. 2.17, 2, Remark 2.30]:

         \mathcal{W}_p(\alpha, \beta)^{p} =  \int_0^1 \left| F^{-1}_{\alpha}(r) - F^{-1}_{\beta}(r) \right|^p dr

        where F^{-1}_{\alpha} is the quantile function, or inverse CDF of \alpha.

        We approximate this integral by a sum of distances between quantiles as it's done in POT [3]:

        \mathcal{W}_p(\alpha, \beta)^{p} =  \sum_{i=1}^n \left| F^{-1}_{\alpha}(r_i) - F^{-1}_{\beta}(r_i) \right|^p (r_i - r_{i-1}), 

        where r_i is the ith quantile of the ordered set of quantiles of \alpha and \beta. We use the step function to compute and inverse the 
            CDF by "holding" the value of the quantile constant between quantiles.

        [1] F. Santambrogio, “Optimal transport for applied mathematicians,” Birkäuser, NY, vol. 55, no. 58–63, p. 94, 2015.
        [2] G. Peyré and M. Cuturi, “Computational optimal transport,” Foundations and Trends in Machine Learning, vol. 11, no. 5–6, pp. 355–607, 2019.
        [3] R. Flamary et al., “POT: Python optimal transport,” Journal of Machine Learning Research, vol. 22, no. 78, pp. 1–8, 2021.

        Code inspired by POT's implementation: https://pythonot.github.io/_modules/ot/lp/solver_1d.html#wasserstein_1d

        Args:
            u_values: Tensor of shape (batch, n) containing the locations of weights values of the first distribution.
            u_weights: Tensor of shape (batch, n) containing the weights of the first distribution.
            v_values: Tensor of shape (batch, m) containing the locations of weights values of the second distribution.
            v_weights: Tensor of shape (batch, m) containing the weights of the second distribution.
            p: Order of the Wasserstein distance.
            require_sort: If True, sort u_values and v_values before computing the loss.
            return_quantiles: If True, return the quantiles of u and v.
            limit_quantile_range: If True, set the distance to 0 if the quantile is greater than 1 (which can happen if non-normalized weights are used as input).
        Returns:
            The 1D Wasserstein distance between the two distributions.

    """

    assert p >= 1, f"The OT loss is only valid for p>=1, {p} was given"

    n = u_values.shape[1]
    m = v_values.shape[1]

    if u_weights is None:
        u_weights = torch.full(
            u_values.shape, 1.0 / n, device=u_values.device, dtype=u_values.dtype
        )

    if v_weights is None:
        v_weights = torch.full(
            v_values.shape, 1.0 / m, device=v_values.device, dtype=v_values.dtype
        )

    if require_sort:
        u_values, u_sorter = torch.sort(u_values, 1)
        v_values, v_sorter = torch.sort(v_values, 1)
        u_weights = torch.gather(u_weights, 1, u_sorter)
        v_weights = torch.gather(v_weights, 1, v_sorter)

    u_cumweights = torch.cumsum(u_weights, 1)
    v_cumweights = torch.cumsum(v_weights, 1)

    qs = torch.sort(torch.cat((u_cumweights, v_cumweights), 1), 1)[0]
    # qs = torch.sort(torch.concatenate((u_cumweights, v_cumweights), 1), 1)
    u_quantiles = quantile_function(qs, u_cumweights, u_values)
    v_quantiles = quantile_function(qs, v_cumweights, v_values)
    if return_quantiles:
        return u_quantiles, v_quantiles, qs, u_cumweights, v_cumweights
    qs = torch.nn.functional.pad(qs, pad=(1, 0))

    # qs = torch.nn.functional.pad(qs, (1, 0), mode='constant', value=0)
    delta = qs[..., 1:] - qs[..., :-1]
    # Set to 0 if qs > 1
    if limit_quantile_range:
        delta = torch.where(qs[..., 1:] > 1, torch.zeros_like(delta), delta)

    diff_quantiles = torch.abs(u_quantiles - v_quantiles)

    if p == 1:
        return torch.sum(delta * diff_quantiles, 1)
    return torch.sum(delta * diff_quantiles.pow(p), 1)


class Wasserstein1DWithTransform(torch.nn.Module):
    def __init__(
        self,
        p=1,
        fixed_x=None,
        require_sort=True,
        log_scaled_x=False,
        transform_kwargs=None,
        **kwargs,
    ):
        super().__init__()

        self.wasserstein = Wasserstein1D(
            p=p,
            fixed_x=fixed_x,
            require_sort=require_sort,
            log_scaled_x=log_scaled_x,
            **kwargs,
        )
        self.transform = get_transform(transform_kwargs, sample_rate=transform_kwargs.pop("sr", 16000))

    def forward(self, x, y, **kwargs):
        x = self.transform(x)
        y = self.transform(y)
        x_pos = torch.tensor(self.transform.get_frequencies()).to(x.device)
        x_pos = x_pos / x_pos.max()
        y_pos = x_pos.clone()
        return self.wasserstein(x, y, x_pos=x_pos, y_pos=y_pos, **kwargs)


class MixOfLosses(torch.nn.Module):
    def __init__(self, losses, weights=None):
        """Mix of losses.
        Args:
          losses: List of loss functions.
          weights: List of weights for each loss function.
        """
        super().__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, x, y, **kwargs):
        loss = {}
        for loss_fn, weight in zip(self.losses, self.weights):
            loss_ = loss_fn(x, y, **kwargs) * weight
            loss[loss_fn.__class__.__name__] = loss_
        return loss


class MSSLoss(torch.nn.Module):
    """Multi-scale spectrogram loss.
    This loss is the bread-and-butter of comparing two audio signals. It offers
    a range of options to compare spectrograms, many of which are redunant, but
    emphasize different aspects of the signal. By far, the most common comparisons
    are magnitudes (mag_weight) and log magnitudes (logmag_weight).
    """

    def __init__(
        self,
        fft_sizes=(2048, 1024, 512, 256, 128, 64),
        loss_type="L1",
        mag_weight=0.0,
        logmag_weight=0.0,
    ):
        """Constructor, set loss weights of various components.
        Args:
          fft_sizes: Compare spectrograms at each of this list of fft sizes. Each
            spectrogram has a time-frequency resolution trade-off based on fft size,
            so comparing multiple scales allows multiple resolutions.
          loss_type: One of 'L1', 'L2', (or 'COSINE', not implemented in PyTorch).
          mag_weight: Weight to compare linear magnitudes of spectrograms. Core
            audio similarity loss. More sensitive to peak magnitudes than log
            magnitudes.
          logmag_weight: Weight to compare log magnitudes of spectrograms. Core
            audio similarity loss. More sensitive to quiet magnitudes than linear
            magnitudes.
        """
        super().__init__()
        self.fft_sizes = fft_sizes
        self.loss_type = loss_type
        self.mag_weight = mag_weight
        self.logmag_weight = logmag_weight

        self.spectrogram_ops = []
        for size in self.fft_sizes:
            spectrogram_op = functools.partial(compute_mag, size=size, add_in_sqrt=1e-10)
            self.spectrogram_ops.append(spectrogram_op)

    def forward(self, target_audio, audio, **kwargs):
        loss = 0.0

        # Compute loss for each fft size.
        for i, loss_op in enumerate(self.spectrogram_ops):
            target_mag = loss_op(target_audio)
            value_mag = loss_op(audio)

            # Add magnitude loss.
            if self.mag_weight > 0:
                loss += self.mag_weight * mean_difference(
                    target_mag, value_mag, self.loss_type, dims=kwargs.get("dims", None)
                )

            # Add logmagnitude loss, reusing spectrogram.
            if self.logmag_weight > 0:
                target = safe_log(target_mag)
                value = safe_log(value_mag)
                loss += self.logmag_weight * mean_difference(
                    target, value, self.loss_type, dims=kwargs.get("dims", None)
                )
        return loss
