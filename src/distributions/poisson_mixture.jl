##### Poisson Mixture distributions #####
"""Upper truncation of the model CCDFs and of the convoluted pmf.

Note:
P(X >= k) is summed up to `CCDF_K_MAX`, so the mass above it is lost. For a Lomax
tail of index α the relative deficit at k is about (k / CCDF_K_MAX)^α: with the
CoMix2 non-home fit (α ≈ 1.24) it is 2.4% at k = 10^4 (-0.011 in log10), which is
invisible on the CCDF panels, against 42% with the earlier 20_000.
Do not exceed ~5e5: `quadgk` in `logpdf(::PoissonLomax, k)` misses the peak of
the integrand at k = 10^6.
"""
const CCDF_K_MAX = 200_000

abstract type PoissonMixture <: DiscreteUnivariateDistribution end

Base.@kwdef struct NegBin <: PoissonMixture
	m::Real
	k::Real
end

Base.@kwdef struct PoissonLogNormal <: PoissonMixture
	μ::Real
	σ::Real
end

Base.@kwdef struct PoissonLomax <: PoissonMixture
	α::Real # shape
	θ::Real # scale
end

# Memoized because logpdf of a PoissonMixture is a numerical integral.
@memoize Distributions.pdf(d::PoissonMixture, k::Int64) = exp(logpdf(d, k))

"""Tail sums `s[k+1] = Σ_{j=k}^{k_max} P(X = j)` for k in 0:k_max.

Note:
Accumulated from k_max downward, the same order as the earlier loop over the
memoized `pdf`, so the values are identical. One vector per distribution is
cached instead of k_max + 1 memoized `pdf` entries (~1.6 MB against ~30 MB at
k_max = 200_000). `logpdf` is called directly so the `pdf` cache is not filled.
"""
@memoize function tail_sums(d::PoissonMixture, k_max::Int64)::Vector{Float64}
	# An explicit loop: `cumsum` on floats sums pairwise, which differs in the last bit.
	s = Vector{Float64}(undef, k_max + 1)
	s[k_max+1] = exp(logpdf(d, k_max))
	for j in (k_max-1):-1:0
		s[j+1] = s[j+2] + exp(logpdf(d, j))
	end
	return s
end

function Distributions.ccdf(d::PoissonMixture, k::Int64; k_max = CCDF_K_MAX)
	if k > k_max
		error("Increase k_max")
	end
	return tail_sums(d, k_max)[k+1]
end

mean100(d::PoissonMixture) = sum(k * pdf(d, k) for k in 0:100)
var100(d::PoissonMixture) = sum( (k-mean(d))^2 * pdf(d, k) for k in 0:100)

##### Negative Binomial distribution #####
Distributions.mean(d::NegBin) = d.m
Distributions.cov(d::NegBin) = sqrt(1/d.m + 1/d.m + 1/d.k)
Distributions.var(d::NegBin) = d.m + d.m^2/d.k

function Distributions.logpdf(d::NegBin, y::Int64)
	return loggamma(d.k+y) - loggamma(d.k) - loggamma(y+1) +
		   d.k*log(d.k/(d.m+d.k)) + y*log(d.m/(d.m+d.k))
end

function Distributions.rand(d::NegBin, n::Int64 = 1)
	negbin = NegativeBinomial(d.k, d.k/(d.m + d.k))
	return rand(negbin, n)
end
Distributions.rand(d::NegBin) = rand(d, 1)[1]

##### Piosson-Lognormal distributions #####
Distributions.mean(d::PoissonLogNormal) = exp(d.μ + d.σ^2 / 2)
Distributions.cov(d::PoissonLogNormal) = sqrt(1 / mean(d) + exp(d.σ^2) - 1)
Distributions.var(d::PoissonLogNormal) = mean(d) + mean(d)^2 * (exp(d.σ^2) - 1)

function Distributions.rand(d::PoissonLogNormal, n::Int64 = 1)
	lognorm = LogNormal(d.μ, d.σ)
	return rand.(Poisson.(rand(lognorm, n)))
end
Distributions.rand(d::PoissonLogNormal) = rand(d, 1)[1]

function Distributions.logpdf(d::PoissonLogNormal, k::Int64)
	@unpack μ, σ = d
	function integrand(x)
		l_int = -loggamma(k+1) + (k-1)*log(x) - x - (log(x)-μ)^2/(2σ^2)
		return exp(l_int)
	end
	# Avoid the integrand to be 0.
	k_tmp = k == 0 ? 1 : k
	int, err = quadgk(u -> integrand(u * k_tmp) * k_tmp, 1e-6, Inf, rtol = 1e-8)
	ret = log(int * 1/(σ*√(2π)))
	return ret
end

##### Lomax distribution #####
struct Lomax <: ContinuousUnivariateDistribution
	α::Real # shape
	θ::Real # scale
end
Distributions.rand(d::Lomax) = (1 / rand() ^ (1/d.α) - 1) * d.θ
Distributions.rand(rng::AbstractRNG, d::Lomax) = rand(rng)
Distributions.minimum(d::Lomax) = 0.0
Distributions.maximum(d::Lomax) = Inf

function Distributions.logpdf(d::Lomax, x::Real)::Real
	@unpack α, θ = d
	if x < 0
		return -Inf
	else
		return log(α) - log(θ) - (α+1)*log(1+x/θ)
	end
end
Distributions.pdf(d::Lomax, x::Real) = exp(logpdf(d, x))
Distributions.ccdf(d::Lomax, x::Real) = (1 + x/d.θ)^(-d.α)
Distributions.cdf(d::Lomax, x::Real) = 1 - ccdf(d, x)

##### Poisson-Lomax distributions #####
Distributions.mean(d::PoissonLomax) = d.α > 1 ? d.θ / (d.α-1) : NaN
function Distributions.var(d::PoissonLomax)
	@unpack α, θ = d
	if α <= 1
		return NaN
	elseif α <= 2
		return Inf
	else
		return mean(d) + θ^2 * α / (α - 1)^2 / (α - 2)
	end
end

function Distributions.cov(d::PoissonLomax)
	@unpack α, θ = d
	if α <= 1
		return NaN
	elseif α <= 2
		return Inf
	else
		return √(1 / mean(d) + α / (α - 2))
	end
end

function Distributions.rand(d::PoissonLomax, n::Int64 = 1)
	@unpack α, θ = d
	y = rand(n)
	x = (1 ./ y .^ (1/α) .- 1) .* θ
	return rand.(Poisson.(x))
end
Distributions.rand(d::PoissonLomax) = rand(d, 1)[1]

function Distributions.logpdf(d::PoissonLomax, k::Int64)
	@unpack α, θ = d
	function integrand(λ)
		l_int = k*log(λ) - λ - loggamma(k+1) + log(α) + α*log(θ) - (α+1)*log(λ+θ)
		return (exp(l_int))
	end
	# Avoid the integrand to be 0.
	k_tmp = k == 0 ? 1 : k
	int, err = quadgk(u -> integrand(u * k_tmp) * k_tmp, 1e-6, Inf, rtol = 1e-8)
	return log(int)
end
Distributions.cdf(d::PoissonLomax, k::Int64)::Real = sum(pdf(d, i) for i in 0:k)
