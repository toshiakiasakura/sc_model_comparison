##### Adhoc functions #####
function exponential_convert(log_p1::Real, log_p2::Real)
	return (exp(log_p1), exp(log_p2))
end

function PoissonLogNormal_convert(μ_obs_ln::AbstractVector, log_σ_ln::AbstractVector)
	σ_ln = exp.(log_σ_ln)
	μ_ln = μ_obs_ln .- σ_ln.^2 ./ 2
	return (μ_ln, σ_ln)
end

is_outside_cutoff_PoissonLogNormal(μ_obs_ln::Real, log_σ_ln::Real) =
	abs(log_σ_ln) > 5.0 || abs(μ_obs_ln) > 5.0
is_outside_cutoff_PoissonLomax(log_α_lo::Real, log_β_lo::Real)::Bool =
	abs(log_α_lo) > 10 || abs(log_β_lo) > 10




##### General Turing.jl helper functions #####
function extract_chain_info(chn::Chains)
	r1 = summarize(chn, mean, median, var, std, ess, rhat) |> DataFrame
	r2 = hpd(chn) |> DataFrame # 95% HPD interval
	r = innerjoin(r1, r2, on = :parameters)
	r = @chain r begin
		@transform :parameters = String.(:parameters)
		@transform :xerr_l = :mean - :lower
		@transform :xerr_u = :upper - :mean
	end
	return r
end

"""Check a Chinas object is converged.
"""
function is_chains_converged(chn::Chains)::Bool
	res = summarystats(chn)[:, [:ess_bulk, :rhat]] |> DataFrame
	ess_flag = all(res[:, :ess_bulk] .> 200)
	rhat_flag = all(res[:, :rhat] .< 1.1)
	cond = (ess_flag == true) & (rhat_flag == true)
	return cond
end

function explore_chns_general(chn::Chains)
	df_res = extract_chain_info(chn);
	@show is_chains_converged(chn)
	@subset(df_res, :ess .< 200) |> display
	# Check the likelihood behaviour
	plot(chn, [:lp, :loglikelihood, :logprior, :tree_depth, :acceptance_rate]) |> display
	plot(chn) |> display
end