##### Adhoc functions #####
function exponential_convert(log_p1::Real, log_p2::Real)
	return (exp(log_p1), exp(log_p2))
end

function exponential_convert(log_p1::AbstractVector, log_p2::AbstractVector)
	p1 = exp.(log_p1)
	p2 = exp.(log_p2)
	return (p1, p2)
end

function PoissonLogNormal_convert(μ_obs_ln::Real, log_σ_ln::Real)
	σ_ln = exp(log_σ_ln)
	μ_ln = μ_obs_ln - σ_ln^2 / 2
	return (μ_ln, σ_ln)
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

function calculate_loglikelihood(dd::DegreeDist, d::DiscreteUnivariateDistribution)
	return sum(logpdf.(d, dd.x) .* dd.y)
end

get_dist_name_from_model(model::Function)::String = String(Symbol(model))[7:end]

function get_parms_single_individual(
	chn::Chains, c1::Symbol, c2::Symbol, c3::Symbol, convert_func::Function)
	med_parms = Dict(p => median(chn[p]) for p in names(chn, :parameters));
	p1, p2, π0 = getindex.(Ref(med_parms), (c1, c2, c3))
	p1, p2 = convert_func(p1, p2)
	return (p1, p2, π0)
end

function get_ZeroInfNegativeBinomial(chn::Chains)
	p1, p2, π0 = get_parms_single_individual(
		chn, :log_m_ga, :log_k_ga, :π0,
		exponential_convert)
	return ZeroInfDist(π0, NegBin(p1, p2))
end

function get_ZeroInfPoissonLogNormal(chn::Chains)
	p1, p2, π0 = get_parms_single_individual(
		chn, :μ_obs_ln, :log_σ_ln, :π0,
		PoissonLogNormal_convert)
	return ZeroInfDist(π0, PoissonLogNormal(p1, p2))
end

function get_ZeroInfPoissonLomax(chn::Chains)
	p1, p2, π0 = get_parms_single_individual(
		chn, :log_α_lo, :log_β_lo, :π0,
		exponential_convert)
	return ZeroInfDist(π0, PoissonLomax(p1, p2))
end

function get_ZeroInfDist(chn::Chains, key::String)
	if key == "ZeroInfNegativeBinomial"
		return get_ZeroInfNegativeBinomial(chn)
	elseif key == "ZeroInfPoissonLogNormal"
		return get_ZeroInfPoissonLogNormal(chn)
	elseif key == "ZeroInfPoissonLomax"
		return get_ZeroInfPoissonLomax(chn)
	else
		error("not applicable: ", model_name)
	end
end

function get_ZeroInfConvDist(chn::Chains, dd_hm::DegreeDist)
	med_parms = Dict(p => median(chn[p]) for p in names(chn, :parameters));
	p1_ln, p2_ln, π0_ln = getindex.(Ref(med_parms), (:μ_obs_ln, :log_σ_ln, :π0_ln))
	p1_lo, p2_lo, π0_lo = getindex.(Ref(med_parms), (:log_α_lo, :log_β_lo, :π0_lo))

	μ_ln, σ_ln = PoissonLogNormal_convert(p1_ln, p2_ln)
	α_lo, β_lo = exponential_convert(p1_lo, p2_lo)

	d_hm = ZeroInfDist(π0_ln, PoissonLogNormal(μ_ln, σ_ln))
	d_nhm = ZeroInfDist(π0_lo, PoissonLomax(α_lo, β_lo))
	conv_dist = ZeroInfConvolutedDist(d_hm, d_nhm, maximum(dd_hm))
	return conv_dist
end

function get_vec_ZeroInfDist_from_chn(chn::Chains, key::String)
	samples = get(chn, chn.name_map.parameters)
	p1 = samples[1].data[:, 1]
	p2 = samples[2].data[:, 1]
	p3 = samples[3].data[:, 1]

	if key == "ZeroInfNegativeBinomial"
		p1, p2 = exponential_convert(p1, p2)
		return ZeroInfDist.(p3, NegBin.(p1, p2))
	elseif key == "ZeroInfPoissonLogNormal"
		p1, p2 = PoissonLogNormal_convert(p1, p2)
		return ZeroInfDist.(p3, PoissonLogNormal.(p1, p2))
	elseif key == "ZeroInfPoissonLomax"
		p1, p2 = exponential_convert(p1, p2)
		return ZeroInfDist.(p3, PoissonLomax.(p1, p2))
	else
		error("Unknown distribution: $key")
	end
end

#############################################
##### Fractional multinomial regression #####
#############################################
"""
Args:
- df_ana: DataFrame after `prepare_ana_for_fmnl`
"""
function one_hot_encoding_multi_vars(df_ana::DataFrame)
	df_ana[:, :y_dummy] .= 1
	f = @formula(y_dummy ~  1 + log10(n_sample) + group_c + mode_cate + cutoff_less90)
	f = apply_schema(f, schema(f, df_ana))
	f |> display
	resp, pred = modelcols(f, df_ana)
	_, x_names = coefnames(f);
	Y = df_ana[:, model_names] |> Matrix{Float64};
	return (pred, Y, x_names)
end

function get_β_med(chn::Chains, n_x::Int)
	chn_res = extract_chain_info(chn)
	β1_med = chn_res[1:n_x, :median]
	β2_med = chn_res[(n_x+1):(2*n_x), :median]
	return (β1_med, β2_med)
end

# TODO: df_obs is needed
function pred_fmnl_multi_vars(chn::Chains, pred::Matrix)
	n_x = size(pred, 2)
	β1_med, β2_med = get_β_med(chn, n_x)
	pred_waic = calculate_fmnl_probs(pred, β1_med, β2_med)
	df_pred = DataFrame(pred_waic, model_names)
	df_pred_cum = create_tab_cum(df_pred, model_names);
	df_pred_cum[:, :key] = df_ana[:, :key];
	df_all_obs = @subset(df_obs, :strat .== "all")
	df_pred_cum = @pipe leftjoin(df_pred_cum, df_all_obs, on = :key) |>
						sort(_, :n_answer; rev = false);
	return df_pred_cum
end


##############################################
##### General Turing.jl helper functions #####
##############################################
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

function explore_chns(chn::Chains)
	df_res = extract_chain_info(chn);
	@show is_chains_converged(chn)
	@subset(df_res, :ess .< 200) |> display
	# Check the likelihood behaviour
	plot(chn, [:lp, :loglikelihood, :logprior, :tree_depth, :acceptance_rate]) |> display
	plot(chn) |> display
end