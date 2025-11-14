function fit_model_with_forward_mode(model, n_sample; iparms = Dict(), progress = true)
	Random.seed!(1236)
	sampler = NUTS()
	@time chn = sample(model, sampler, n_sample;
		progress = progress, initial_params = NamedTuple(iparms))
end

@model function model_ZeroInfNegativeBinomial(dd::DegreeDist)
	log_m_ga ~ Normal(0.0, 2)
	log_k_ga ~ Normal(0.0, 1)
	m = exp(log_m_ga)
	k = exp(log_k_ga)
	π0 ~ Beta(1.5, 1.5)
	zeroInf = ZeroInfDist(π0, NegBin(m, k))
	ll = calculate_loglikelihood(dd, zeroInf)
	Turing.@addlogprob! ll
end

@model function model_ZeroInfPoissonLogNormal(dd::DegreeDist)
	μ_obs_ln ~ Normal(0, 1.0)
	log_σ_ln ~ Normal(0, 1.0)
	π0 ~ Beta(1.5, 1.5)
	cond = is_outside_cutoff_PoissonLogNormal(μ_obs_ln, log_σ_ln)
	if cond == true
		Turing.@addlogprob! -Inf # Reject sample point immediately
		return
	end

	μ_ln, σ_ln = PoissonLogNormal_convert(μ_obs_ln, log_σ_ln)
	zeroInf = ZeroInfDist(π0, PoissonLogNormal(μ_ln, σ_ln))
	ll = calculate_loglikelihood(dd, zeroInf)
	Turing.@addlogprob! ll
end

@model function model_ZeroInfPoissonLomax(dd::DegreeDist)
	log_α_lo ~ Normal(0, 1.0)
	log_β_lo ~ Normal(0, 2.0)
	α = exp(log_α_lo)
	β = exp(log_β_lo)
	π0 ~ Beta(1.5, 1.5)
	cond = is_outside_cutoff_PoissonLomax(log_α_lo, log_β_lo)
	if cond == true
		Turing.@addlogprob! -Inf # Reject sample point immediately
		return
	end
	zeroInf = ZeroInfDist(π0, PoissonLomax(α, β))
	ll = calculate_loglikelihood(dd, zeroInf)
	Turing.@addlogprob! ll
end

@model function model_ZeroInfConvDist(dd_all::DegreeDist, dd_hm::DegreeDist, prior_dic)
	μ_obs_ln ~ prior_dic["hm_p1"]
	log_σ_ln ~ prior_dic["hm_p2"]
	π0_ln ~ Beta(1.5, 1.5)

	log_α_lo ~ prior_dic["nhm_p1"]
	log_β_lo ~ prior_dic["nhm_p2"]
	π0_lo ~ Beta(1.5, 1.5)

	μ_ln, σ_ln = PoissonLogNormal_convert(μ_obs_ln, log_σ_ln)
	α_lo, β_lo = exponential_convert(log_α_lo, log_β_lo)

	d_hm = ZeroInfDist(π0_ln, PoissonLogNormal(μ_ln, σ_ln))
	d_nhm = ZeroInfDist(π0_lo, PoissonLomax(α_lo, β_lo))
	conv_dist = ZeroInfConvolutedDist(d_hm, d_nhm, maximum(dd_hm))
	ll = calculate_loglikelihood(dd_all, conv_dist)
	Turing.@addlogprob! ll
end