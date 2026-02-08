##### Poisson Mixture distributions #####
abstract type PoissonMixture <: DiscreteUnivariateDistribution end
Base.length(p::PoissonMixture)=1
Base.iterate(p::PoissonMixture) = (p, nothing)
Base.iterate(p::PoissonMixture, nothing) = nothing

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

@memoize function Distributions.ccdf(
    d::PoissonMixture,
    k::Int64;
    k_max=20_000
)
    if k > k_max
        error("Increase k_max")
	elseif k == k_max
		return pdf(d, k)
	else
		return Distributions.ccdf(d, k+1; k_max=k_max) + pdf(d, k)
    end
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


###### Double Pareto lognormal (dPlN) continuous distribution #####
# Reed & Jorgensen (2004)
struct DPlN <: ContinuousUnivariateDistribution
	α::Real # right tail exponent (controls upper power-law tail: x^{-α-1})
	β::Real # left tail exponent (controls lower power-law tail: x^{β-1})
	ν::Real # location (mode of lognormal body)
	τ::Real # scale (std dev of lognormal body)
end
Distributions.minimum(d::DPlN) = 0.0
Distributions.maximum(d::DPlN) = Inf

function Distributions.logpdf(d::DPlN, x::Real)::Real
	@unpack α, β, ν, τ = d
	if x <= 0
		return -Inf
	end
	logx = log(x)
	log_common = log(α) + log(β) - log(α + β)
	log_comp1 = log_common + (-α - 1) * logx + α * ν + α^2 * τ^2 / 2 +
	            logcdf(Normal(), (logx - ν - α * τ^2) / τ)
	log_comp2 = log_common + (β - 1) * logx - β * ν + β^2 * τ^2 / 2 +
	            logcdf(Normal(), (logx - ν + β * τ^2) / τ)
	return logaddexp(log_comp1, log_comp2)
end
Distributions.pdf(d::DPlN, x::Real) = exp(logpdf(d, x))

function Distributions.rand(d::DPlN)
	@unpack α, β, ν, τ = d
	# Generative process: X = exp(Y - W1 + W2)
	# where Y ~ N(ν,τ²), W1 ~ Exp(rate=α), W2 ~ Exp(rate=β)
	Y = rand(Normal(ν, τ))
	W1 = rand(Exponential(1 / α))
	W2 = rand(Exponential(1 / β))
	return exp(Y - W1 + W2)
end

function Distributions.mean(d::DPlN)
	@unpack α, β, ν, τ = d
	# E[X] = αβ/((α-1)(β+1)) * exp(ν + τ²/2) for α > 1
	α > 1 ? α * β / ((α - 1) * (β + 1)) * exp(ν + τ^2 / 2) : NaN
end

###### Poisson-double Pareto lognormal distribution #####
Base.@kwdef struct PoissonDPlN <: PoissonMixture
	α::Real # right tail exponent
	β::Real # left tail exponent
	ν::Real # location
	τ::Real # scale
end

function Distributions.mean(d::PoissonDPlN)
	@unpack α, β, ν, τ = d
	mean(DPlN(α, β, ν, τ))
end

function Distributions.var(d::PoissonDPlN)
	@unpack α, β, ν, τ = d
	# Var[K] = E[λ] + Var[λ] = E[λ] + E[λ²] - E[λ]²
	# E[X^r] = αβ/((α-r)(β+r)) * exp(rν + r²τ²/2)
	if α <= 1
		return NaN
	elseif α <= 2
		return Inf
	else
		m1 = mean(d)
		m2 = α * β / ((α - 2) * (β + 2)) * exp(2 * ν + 2 * τ^2)
		return m1 + m2 - m1^2
	end
end

function Distributions.cov(d::PoissonDPlN)
	@unpack α, β, ν, τ = d
	if α <= 1
		return NaN
	elseif α <= 2
		return Inf
	else
		m1 = mean(d)
		m2 = α * β / ((α - 2) * (β + 2)) * exp(2 * ν + 2 * τ^2)
		return sqrt(1 / m1 + (m2 - m1^2) / m1^2)
	end
end

function Distributions.rand(d::PoissonDPlN, n::Int64 = 1)
	dpln = DPlN(d.α, d.β, d.ν, d.τ)
	return [rand(Poisson(rand(dpln))) for _ in 1:n]
end
Distributions.rand(d::PoissonDPlN) = rand(d, 1)[1]

function Distributions.logpdf(d::PoissonDPlN, k::Int64)
	dpln = DPlN(d.α, d.β, d.ν, d.τ)
	function integrand(λ)
		l_int = k * log(λ) - λ - loggamma(k + 1) + logpdf(dpln, λ)
		return exp(l_int)
	end
	# Avoid the integrand to be 0.
	k_tmp = k == 0 ? 1 : k
	int, err = quadgk(u -> integrand(u * k_tmp) * k_tmp, 1e-6, Inf, rtol = 1e-8)
	return log(int)
end
Distributions.cdf(d::PoissonDPlN, k::Int64)::Real = sum(pdf(d, i) for i in 0:k)

#####################################
###### Mixture distributions ########
#####################################
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
