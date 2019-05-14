import torch
import torch.distributions as dist

class NormalMixture(dist.Distribution):
    arg_constraints = {'locs': dist.constraints.real, 'scales': dist.constraints.positive}
    support = dist.constraints.real
    has_rsample = True

    def __init__(self, locs, scales, weights=None):
        self.locs, self.scales = dist.utils.broadcast_all(locs, scales)
        self.component_shape = self.locs.size(0)
        self.weights = weights if weights is not None else torch.ones(locs.size(0), device=locs.device) / locs.size(0)
        self._batch_shape = self.locs.shape

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
            assignements = dist.Categorical(self.weights).sample(sample_shape)
            # unsqueeze(1) to get similar shape (N, 1, D) than with standard Gaussian
            return dist.Normal(self.locs[assignements].unsqueeze(1), self.scales[assignements].unsqueeze(1)).rsample()

    def log_prob(self, value):
        # value of shape (K, B, D)
        flat_rest = value.shape[:2]  # K, B
        xs = value.contiguous().view(torch.Size([-1, *value.shape[2:]]))
        log_probs = dist.Normal(self.locs.unsqueeze(1), self.scales.unsqueeze(1)).log_prob(xs.unsqueeze(0)).sum(-1)
        log_w = self.weights.log().view(log_probs.size(0), 1)
        prob = torch.logsumexp(log_w + log_probs, 0)
        return prob.view(*value.shape[:2], -1)
