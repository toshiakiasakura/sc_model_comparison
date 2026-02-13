
###############################
##### Simulation analysis #####
###############################

function get_CoMix2_fitted_dists()
	model_names = get_model_names()
	res = load("../dt_intermediate/CoMix2_chns.jld2")["result"]
	chns_hm = res["chns_home"]
	chns_nhm = res["chns_non-home"]
	dists_hm = [get_ZeroInfDist(chns_hm[m], m) for m in model_names]
	dists_nhm = [get_ZeroInfDist(chns_nhm[m], m) for m in model_names]
	return (dists_hm, dists_nhm)
end

function get_best_ZInf_mean(strat::String)
	dists_hm, dists_nhm = get_CoMix2_fitted_dists();
	m = strat == "home" ? mean(dists_hm[2]) : mean(dists_nhm[3])
	return m
end

function get_ZInf_means(strat::String)
	dists_hm, dists_nhm = get_CoMix2_fitted_dists();
	ms = strat == "home" ? mean.(dists_hm) : mean.(dists_nhm)
	return ms
end

function simulate_and_take_sample_mean(strat)
	dd_all, dd_hm, dd_nhm = get_comix2_dd_all_hm_nhm()
	dists_hm, dists_nhm = get_CoMix2_fitted_dists();

	# choose data according to `strat`
	dd = strat == "home" ? dd_hm : dd_nhm
	dists = strat == "home" ? dists_hm : dists_nhm

	N = 100
	df_mer = DataFrame()
	M_lis = [100, 1000, 10000, 37347]
	for M in M_lis
		boot_sim = [rand(dd, M) |> mean for _ in 1:N]
		df_mer = vcat(df_mer,
			DataFrame(:mean => boot_sim, :sample_size => M, :tp => "Bootstrap"))
		for (m, d) in zip(["ZInf-NB", "ZInf-PLN", "ZInf-PLomax"], dists)
			sim = [rand(d, M) |> mean for _ in 1:N]
			df_mer = vcat(df_mer,
				DataFrame(:mean => sim, :sample_size => M, :tp => m))
		end
	end
	return df_mer
end

# Fit data bootstrapped from fitted distribution with three models.
function fit_bootstrap_data_CoMix2(M_lis; rep = 10)
	dists_hm, dists_nhm = get_CoMix2_fitted_dists();
	dist_LN_hm = dists_hm[2]
	dist_PLomax_nhm = dists_nhm[3]
	for M in M_lis
		res_mer = Vector{Any}(undef, rep)  # preallocate to avoid push! allocations
		Threads.@threads for r in 1:rep
			dds = Dict(
				"home" => rand(dist_LN_hm, M) |> DegreeDist,
				"non-home" => rand(dist_PLomax_nhm, M) |> DegreeDist,
			)
			res_mer[r] = fit_hm_nhm_dds(dds)
		end
		path = "../dt_intermediate_bootstrap/CoMix2_$(M)samples_$(rep)repeat.jld2"
		jldsave(path, result = res_mer)
	end
end

function parse_bootstrap_estimated_data(paths, strat, M_lis)
	m_PLomax = get_best_ZInf_mean(strat)
	model_names = get_model_names()

	df_sum = DataFrame()
	for (i, path) in enumerate(paths)
		M = M_lis[i]
		res = load(path)["result"];
		N = length(res)
		for m in model_names
			chns = [res[i]["chns_$(strat)"][m] for i in 1:N]
			ms = [get_ZeroInfDist(chns[i], m) |> mean for i in 1:N]
			cov_flag = [
				@pipe get_vec_ZeroInfDist_from_chn(chns[i], m) .|> mean |>
					  (x -> x[.~isnan.(x)]) |>
					  quantile(_, [0.025, 0.975]) |> (x -> (x[1] < m_PLomax < x[2]))
				for i in 1:N
			]
			df_m = DataFrame(mean = ms, sample_size = M, tp = m, cov_flag = cov_flag)
			df_sum = vcat(df_sum, df_m)
		end
		ms = [res[i]["dds"][strat] |> mean for i in 1:N]
		df_m = DataFrame(mean = ms, sample_size = M, tp = "Sample mean", cov_flag = false)
		df_sum = vcat(df_sum, df_m)
	end
	df_sum
end

function plot_illustrative_dist_comparison()
	# Base distributions inferred from the fitted distributions.
	d_negbin = NegBin(4.0, 0.169)
	@show mean100(d_negbin)
	@show var100(d_negbin);
	d_lognormal = PoissonLogNormal(-0.046, 1.84)
	@show mean100(d_lognormal)
	@show var100(d_lognormal);
	d_lomax = PoissonLomax(1.239, 1.759)
	@show mean100(d_lomax)
	@show var100(d_lomax);

	# Simulated data and take means.
	n_sim = 1_00
	n_sample = 1000
	ms_negbin = [mean(rand(d_negbin, n_sample)) for _ in 1:n_sim]
	ms_lognormal = [mean(rand(d_lognormal, n_sample)) for _ in 1:n_sim]
	ms_lomax = [mean(rand(d_lomax, n_sample)) for _ in 1:n_sim]
	df_ms = DataFrame(
		:mean => vcat(ms_negbin, ms_lognormal, ms_lomax),
		:model => vcat(
			fill("NB", n_sim), fill("PLN", n_sim), fill("PLomax", n_sim)))
	df_ms.model = categorical(df_ms.model,
		levels = ["NB", "PLN", "PLomax"]);

	pos = (-0.1, 1.12)
	pl1 = plot(xlim = [0, 30], ylabel = "Probability mass function",
		xlabel = "Number of contacts per day",
		bottom_margin = 5Plots.mm, left_margin = 5Plots.mm, top_margin = 10Plots.mm)
	plot_pdf!(pl1, d_negbin; conv_log10 = false, label = "Negative binomial")
	plot_pdf!(pl1, d_lognormal; conv_log10 = false, label = "Poisson-lognormal")
	plot_pdf!(pl1, d_lomax; conv_log10 = false, label = "Poisson-Lomax")
	annotate!(pl1, pos, text("A", :left, 18, "Helvetica"))

	xtk = ([1, 10, 100, 1000], [L"1", L"10", L"10^{2}", L"10^{3}"])
	ytk = ([-5, -4, -3, -2, -1, 0],
		[L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"])
	pl2 = plot(xlim = [1, 5_000], ylim = [-5, 0], ylabel = "CCDF",
		xlabel = "Number of contacts per day", xticks = xtk, yticks = ytk)
	plot_ccdf!(pl2, d_negbin; conv_log10 = false, label = "Negative binomial")
	plot_ccdf!(pl2, d_lognormal; conv_log10 = false, label = "Poisson-lognormal")
	plot_ccdf!(pl2, d_lomax; conv_log10 = false, label = "Poisson-Lomax")
	annotate!(pl2, pos, text("B", :left, 18, "Helvetica"))

	pl3 = plot(ylim = [0, 20], ylabel = "Sample means of 1000 simulated contacts")
	dotplot!(pl3, df_ms[:, :model], df_ms[:, :mean],
		legend = false, color = 1, markerstrokewidth = 0.1, markersize = 2.2, xrotation = 0)
	boxplot!(pl3, df_ms[:, :model], df_ms[:, :mean], outliers = false, fillalpha = 0.75, color = 1)
	hline!(pl3, [3.909], ls = :dot, color = 2, lw = 2.5)
	annotate!(pl3, pos, text("C", :left, 18, "Helvetica"))

	pl = plot(pl1, pl2, pl3, layout = (1, 3), size = (900, 300), dpi = 300)
	return pl
end

"""
Args:
- df_sum: returned by `parse_bootstrap_estimated_data`.
"""
function plot_coverage_prob(df_sum::DataFrame, strat)
	model_names = get_model_names()
	ms = ["ZInf-NB", "ZInf-PLN", "ZInf-PLomax"]
	rep_dic = Dict(m => m_new for (m, m_new) in zip(model_names, ms))
	df_cov = @pipe groupby(df_sum, [:tp, :sample_size]) |> combine(_) do gdf
					   n = nrow(gdf)
					   n_cov = sum(gdf[:, :cov_flag])
					   (cov_prob = n_cov/n*100,)
				   end |> @subset(_, :tp .!= "Sample mean") |>
				   @transform(_, :tp = replace.(:tp, rep_dic...),
					   :sample_size = string.(:sample_size))
	pl = plot(; xlabel = "Number of samples per simulation",
		ylabel = "Coverage probability (%)", xlabelfontsize = 11, ylabelfontsize = 12,
		ylim = [0, 105],
	)
	plot!(pl,
		df_cov[:, :sample_size], df_cov[:, :cov_prob], group = df_cov[:, :tp],
		marker = :circle, markerstrokewidth = 0.5, color = [7 6 13],
		legend = (0.7, 0.5),
		#xlim=[2.8,6.2],
	)
	hline!(pl, [95], ls = :dash, color = :black, label = "", alpha = 0.7)
	return pl
end

"""
Args:
- df_mer: returned by `simulate_and_take_sample_mean`.
"""
function plot_simulated_sample_mean(df_mer::DataFrame, strat::String; color = 1)
	pls = []
	tps = ["Bootstrap", "ZInf-NB", "ZInf-PLN", "ZInf-PLomax"]
	ms = get_ZInf_means(strat)

	dd_all, dd_hm, dd_nhm = get_comix2_dd_all_hm_nhm()
	dd = strat == "home" ? dd_hm : dd_nhm
	ms = vcat([mean(dd)], ms)

	ylim = strat == "home" ? [0, 3] : [0, 10]
	for (i, tp) in enumerate(tps)
		df_tmp = @subset(df_mer, :tp .== tp)
		ylbl = i == 1 ? "Sample mean of simulated data" : ""
		m = ms[i]
		ytk = i == 1 ? true : false

		pl = plot(; ylim = ylim, xrotation = 30, ylabelfontsize = 11)
		dotplot!(pl, df_tmp[:, :sample_size], df_tmp[:, :mean],
			ylabel = ylbl, xlabel = "", yticks = ytk,
			legend = false, color = color,
			markerstrokewidth = 0.1, markersize = 2.2)
		annotate!(pl, (0.5, 0.1
			), text(tp, :black, :centre, 10, "Helvetica"))
		boxplot!(pl, df_tmp[:, :sample_size], df_tmp[:, :mean],
			color = color, outliers = false, fillalpha = 0.75)
		hline!(pl, [m], ls = :dot, color = 2, lw = 3)
		push!(pls, pl)
	end
	plot!(pls[4], right_margin = 5Plots.mm, bottom_margin = 10Plots.mm)
	annotate!(pls[2], (1.0, -0.27),
		text("Number of samples per simulation",
			:black, :centre, 11, "Helvetica"))
	annotate!(pls[1], (-0.35, 1.15), text("A", :left, 18, "Helvetica"))
	plot!(pls[1], top_margin = 10Plots.mm, left_margin = 5Plots.mm)
	return plot(pls..., layout = (1, 4), size = (800, 400))
end

"""
Args:
- df_sum: returned by `parse_bootstrap_estimated_data`,
"""
function plot_estimated_means(df_sum::DataFrame, strat::String)
	model_names = get_model_names()
	m_PLomax = get_best_ZInf_mean(strat)
	order = ["Sample mean", "ZInf-NB", "ZInf-PLN", "ZInf-PLomax"]
	rep_dic = Dict(m => m_new for (m, m_new) in zip(["Sample mean", model_names...], order))
	@transform!(df_sum, :tp = replace(:tp, rep_dic...));
	# Check NaN
	cond = isnan.(df_sum[:, :mean])
	df_sum[cond, :] #|> display
	df_sum_vis = df_sum[.~cond, :];
	df_sum_vis[!, :sample_size] = string.(df_sum_vis[:, :sample_size])
	df_sum_vis = @subset(df_sum_vis, :tp .!= "Sample mean")

	ylim = strat == "home" ? [0, 3] : [0, 10]
	pl = plot_mean(df_sum_vis, m_PLomax;
		order = order,
		ylabel = "Estimated means",
		color = [7 6 13],
		title = "", xlabelfontsize = 12, ylabelfontsize = 12,
		ylim = ylim)
	return pl
end

function plot_mean(df_vis::DataFrame, h_m::Real;
	order = ["ZInf-NB", "ZInf-LN", "ZInf-Lomax"],
	ylabel = "Sample mean of simulated data",
	color = [1 7 6 13],
	kwds...)
	df_vis = copy(df_vis)
	#order = ["Observed", "ZeroTruncNegativeBinomial", "ZeroTruncPoissonLomax", "Best"]
	df_vis.tp = categorical(df_vis.tp, levels = order, ordered = true)

	pl = plot(ylabel = ylabel,
		xlabel = "Number of samples per simulation"; kwds...)
	groupeddotplot!(pl, df_vis[:, :sample_size], df_vis[:, :mean];
		group = df_vis[:, :tp], label = "", color = color,
		markersize = 2.2, markerstrokewidth = 0.2)
	groupedboxplot!(pl, df_vis[:, :sample_size], df_vis[:, :mean]; group = df_vis[:, :tp],
		fillalpha = 0.75, outliers = false, color = color)
	hline!([h_m], ls = :dash, color = :black, label = "", alpha = 0.7) # , label="Baseline eigenvalue")
	return pl
end

function plot_bootstrap_panels(df_mer, df_sum, strat)
	df_mer[!, :sample_size] = string.(df_mer[:, :sample_size])
	pl1 = plot_simulated_sample_mean(df_mer, strat)
	pl2 = plot_estimated_means(df_sum, strat);
	pl3 = plot_coverage_prob(df_sum, strat);

	annotate!(pl2, (-0.15, 1.07), text("B", :left, 18, "Helvetica"))
	annotate!(pl3, (-0.27, 1.07), text("C", :left, 18, "Helvetica"))
	plot!(pl2, left_margin = 5Plots.mm)

	layout = @layout [a; b c{0.4w}]
	pl = plot(pl1, pl2, pl3, layout = layout, size = (800, 600))
	pl
end

###############################
##### Convoluted analysis #####
###############################

function get_comix2_dd_all_hm_nhm()
	df_dds = CSV.read("../dt_surveys_master/master_dds.csv", DataFrame);
	df_comix2 = @subset(df_dds, :key .== "CoMix2")
	dd_all = @subset(df_comix2, :strat .== "all") |> DegreeDist;
	dd_hm = @subset(df_comix2, :strat .== "home") |> DegreeDist;
	dd_nhm = @subset(df_comix2, :strat .== "non-home") |> DegreeDist;
	return (dd_all, dd_hm, dd_nhm)
end

function fit_convoluted_dist(df_dds::DataFrame)
	dd_all, dd_hm, dd_nhm = get_comix2_dd_all_hm_nhm()

	chn_hm = res["chns_home"][model_names[2]]
	chn_nhm = res["chns_non-home"][model_names[3]]
	df_chn_hm = DataFrame(chn_hm);

	prior_kernel(v::Vector) = KernelDensity.kde(v) |> InterpKDEDistribution
	prior_dic = Dict()
	prior_dic["hm_p1"] = prior_kernel(df_chn_hm[:, :μ_obs_ln])
	prior_dic["hm_p2"] = prior_kernel(df_chn_hm[:, :log_σ_ln])
	prior_dic["hm_p3"] = prior_kernel(log.(df_chn_hm[:, :π0]))
	df_chn_nhm = DataFrame(chn_nhm);
	prior_dic["nhm_p1"] = prior_kernel(df_chn_nhm[:, :log_α_lo])
	prior_dic["nhm_p2"] = prior_kernel(df_chn_nhm[:, :log_β_lo])
	prior_dic["nhm_p3"] = prior_kernel(log.(df_chn_nhm[:, :π0]));

	model = model_ZeroInfConvDist(dd_all, dd_hm, prior_dic)
	chn = sample(model, NUTS(), 2000; progress = true)
	jldsave("../dt_intermediate/CoMix2_convoluted_chns.jld2", result = chn)
end

function plot_conv_fit()
	dd_all, dd_hm, dd_nhm = get_comix2_dd_all_hm_nhm()
	model_names = get_model_names()

	chn_conv = load("../dt_intermediate/CoMix2_convoluted_chns.jld2")["result"]
	res = load("../dt_intermediate/CoMix2_chns.jld2")["result"]
	chns_hm = res["chns_home"]
	chns_nhm = res["chns_non-home"]
	dists_hm = [get_ZeroInfDist(chns_hm[m], m) for m in model_names]
	dists_nhm = [get_ZeroInfDist(chns_nhm[m], m) for m in model_names]
	conv_dist = get_ZeroInfConvDist(chn_conv, dd_hm);

	# TODO: replace it
	# m = model_names[3]
	#chn = load("../dt_intermediate/CoMix2_chns_all.jld2")["result"]["chns_all"][m]
	#dist_all = get_ZeroInfDist(chn, m)
	chn = load("../dt_intermediate/CoMix2_chns_all.jld2")["result"]
	dist_all = get_ZeroInfDist(chn, model_names[3])

	pl1 = plot_hm_nhm(dd_hm, dd_nhm, dists_hm, dists_nhm, plot_pdf!; ylim = [-5, 0], xlim = [0, 50])
	pl2 = plot_hm_nhm(dd_hm, dd_nhm, dists_hm, dists_nhm, plot_ccdf!; ylim = [-5, 0])
	xlab = "Number of contacts per day"
	plot!(pl1, xlabel = xlab, ylabel = "Probability mass function",
		left_margin = 5Plots.mm)
	plot!(pl2, xlabel = xlab, ylabel = "CCDF")
	plot(pl1, pl2, size = (800, 400), bottom_margin = 5Plots.mm)

	pl3 = plot_conv(dd_all, dist_all, conv_dist, plot_pdf!; ylim = [-5, 0], xlim = [0, 50])
	pl4 = plot_conv(dd_all, dist_all, conv_dist, plot_ccdf!; ylim = [-5, 0])
	plot!(pl3, xlabel = xlab, ylabel = "Probability mass function",
		left_margin = 5Plots.mm,
	)
	plot!(pl4, xlabel = xlab, ylabel = "CCDF")

	pos = (-0.2, 1.0)
	annotate!(pl1, pos, text("A", :left, 18, "Helvetica"))
	annotate!(pl2, pos, text("B", :left, 18, "Helvetica"))
	annotate!(pl3, pos, text("C", :left, 18, "Helvetica"))
	annotate!(pl4, pos, text("D", :left, 18, "Helvetica"))
	layout = @layout [a b; c d]
	pl = plot(pl1, pl2, pl3, pl4, layout = layout, size = (800, 800),
		dpi = 300, fig = :png, top_margin = 5Plots.mm)
	return pl
end

function plot_hm_nhm(dd_hm::DegreeDist, dd_nhm::DegreeDist, dists_hm, dists_nhm, plot_func!;
	kwds...,
)
	pl = plot(; kwds...)
	colors = [7, 6, 13]
	color = 2
	kwds_obs = (markersize = 2.5, markerstrokewidth = 0.2)
	kwds_fit = (lw = 1.5, ls = :dash)
	plot_func!(pl, dd_hm, color = color, label = "Home (observed)"; kwds_obs...)
	plot_func!(pl, dists_hm[2], color = color, ls = :dash, label = "Home, Best fitted"; kwds_fit...)
	color = 1
	plot_func!(pl, dd_nhm, color = color, label = "Non-home (observed)"; kwds_obs...)
	plot_func!(pl, dists_nhm[1], color = colors[1], label = "Non-home, ZInf-NB"; kwds_fit...)
	plot_func!(pl, dists_nhm[2], color = colors[2], label = "Non-home, ZInf-PLN"; kwds_fit...)
	plot_func!(pl, dists_nhm[3], color = colors[3], label = "Non-home, ZInf-PLomax"; kwds_fit...)
	return pl
end

function plot_conv(dd_all, dist_all, conv_dist, plot_func!; kwds...)
	kwds_obs = (markersize = 2.5, markerstrokewidth = 0.0)
	kwds_fit = (lw = 1.5, ls = :dash)

	pl = plot(; kwds...)
	plot_func!(pl, dd_all, label = "All (observed)"; kwds_obs...)
	plot_func!(pl, dist_all, label = "Best 2-parameter dist"; color = 13, kwds_fit...)
	plot_func!(pl, conv_dist, label = "Convoluted"; color = :red, kwds_fit...)
	#plot_func!(pl, conv_dist_refit, label="Convoluted (no-fit)"; kwds_fit...)
end

function fit_CoMix2_all()
	dd_all, dd_hm, dd_nhm = get_comix2_dd_all_hm_nhm()
	models = [model_ZeroInfNegativeBinomial, model_ZeroInfPoissonLogNormal,
		model_ZeroInfPoissonLomax]
	res = Dict("chns_all" => Dict())
	for model_func in models
		model = model_func(dd_all)
		med, chn = get_median_parms_from_model(model)
		chn = fit_model_with_forward_mode(model, 2000; iparms = med, progress = false)
		res["chns_all"][get_dist_name_from_model(model_func)] = chn
	end
	return res
end

###############################
##### Sample size effect ######
###############################

"""
Note:
See `create_summary_stat_one_data` for a similar function.
"""
function create_summary_stat_bootstrap()
	paths = glob("../dt_intermediate_bootstrap/*.jld2")

	df_all = DataFrame()
	for path in paths
		println("Processing: ", path)
		m = match(r"comix2_(\d+)samples_(\d+)repeat", path)
		if m === nothing
			continue
		end
		sample_size = parse(Int, m.captures[1])
		n_repeat = parse(Int, m.captures[2])

		# Load the bootstrap results
		res_mer = load(path)["result"]
		# Process each bootstrap iteration
		for (i, res) in enumerate(res_mer)
			for strat in ["home", "non-home"]
				df_tmp = summarise_res_one_strat(res, strat, "CoMix2_bootstrap")
				df_tmp[!, :sample_size] .= sample_size
				df_tmp[!, :iteration] .= i
				df_all = vcat(df_all, df_tmp)
			end
		end
	end
	return df_all
end

function obtain_spline_basis(logX; df = 4)
	@rput logX df
	R"""
	library(splines)
	spline_basis <- ns(logX, df=df)
	"""
	@rget spline_basis
	return spline_basis
end

function predict_spline(old_logX, new_logX, df = 4)
	@rput old_logX new_logX df
	R"""
	library(splines)
	require(stats)
	spline_basis <- ns(logX, df=df)
	new_spline <- predict(spline_basis, new_logX)
	"""
	@rget new_spline
	return new_spline
end

function create_spline_basis_sample_size(df_ana::DataFrame; df = 3)
	logX = log10.(df_ana[:, :n_sample])
	spline_basis = obtain_spline_basis(logX; df = df)
	X = hcat(ones(length(logX)), spline_basis)
	return (X, logX)
end

function pred_fmnl_sample_size(chn::Chains, original_logX::Vector{Float64};
	sample_sizes = exp10.(range(log10(90), log10(100_000), length = 50)),
	df = 4)

	n_pred = length(sample_sizes)
	new_logX = log10.(sample_sizes)

	# Create predictor matrix: intercept + spline basis
	spline_pred = predict_spline(original_logX, new_logX, df)
	X_pred = hcat(ones(n_pred), spline_pred)
	n_x = size(X_pred, 2)

	β1_med, β2_med = get_β_med(chn, n_x)
	pred_probs = calculate_fmnl_probs(X_pred, β1_med, β2_med)

	df_pred = DataFrame(
		logX = new_logX,
		sample_size = sample_sizes,
	)
	df_pred[!, :y1] = pred_probs[:, 1]
	df_pred[!, :y2] = pred_probs[:, 2]
	df_pred[!, :y3] = pred_probs[:, 3]
	return df_pred
end

function fit_pred_df_sample_size_empirical(df_res::DataFrame, DF::Int64)
	df_ana = prep_fmnl_vars(df_res);
	# Create spline basis for sample size
	X_spline, logX_original = create_spline_basis_sample_size(df_ana; df = DF)
	pred, Y, x_names = one_hot_encoding_multi_vars(df_ana);
	chn1 = sample(model_fmnl(X_spline, Y), NUTS(), 2000; progress = false)
	df_pred = pred_fmnl_sample_size(chn1, logX_original; df = DF);
	return df_pred
end

"""
Note:
- Data is from `../dt_intermediate_bootstrap/comix2_waic_weights.csv`
"""
function fit_pred_df_sample_size_boot(df_boot, DF)
	df_boot_tab = unstack(df_boot, :key, :model, :weight_waic)
	df_boot_tab = leftjoin(df_boot_tab,
		unique(df_boot[:, [:key, :sample_size]]), on = :key)

	logX_original = log10.(df_boot_tab[:, :sample_size])
	X_spline = obtain_spline_basis(logX_original; df = DF)
	X_spline = hcat(ones(length(logX_original)), X_spline)
	Y = df_boot_tab[:, 2:4] |> Matrix
	chn2 = sample(model_fmnl(X_spline, Y), NUTS(), 2000; progress = false)
	df_pred_boot = pred_fmnl_sample_size(chn2, logX_original; df = DF);
	return df_pred_boot
end

function plot_sample_size_and_best_model(df_pred, df_mer_nh, df_pred_boot, df_boot)
	xticks_ = ([1, 2, 3, 4, 5], ["10", "100", "1000", "10,000", "100,000"])
	pl1 = plot(xlabel = "", ylabel = "WAIC weight",
		xticks = xticks_, legend = (0.8, 0.5))
	colors = [7, 6, 13]
	colors_reshape = reshape(colors, 1, :)
	model_names = get_model_names()
	for i in 1:3
		plot!(pl1, df_pred[:, :logX], df_pred[:, Symbol("y$(i)")],
			label = model_abbr[model_names[i]],
			color = colors[i], lw = 2.0)
	end
	@with df_mer_nh scatter!(pl1, log10.(:n_sample), :weight_waic, group = :model_abbr,
		colour = colors_reshape, label = "",
		markerstrokewidth = 0.4)

	pl2 = plot(xlabel = "sample size", ylabel = "WAIC weight",
		xticks = xticks_, legend = nothing)
	for i in 1:3
		model_name = model_names[i]
		plot!(pl2, df_pred_boot[:, :logX], df_pred_boot[:, Symbol("y$(i)")],
			label = "$(model_name), bootstrap",
			color = colors[i], lw = 2)
	end
	# Add jitter to x-positions for each model to avoid overlap
	offset_scale = 0.05
	for (i, model_name) in enumerate(model_names)
		df_model = @subset(df_boot, :model .== model_name)
		offset = (i - 2) * offset_scale
		scatter!(pl2, log10.(df_model[:, :n_sample]) .+ offset, df_model[:, :weight_waic],
			color = colors[i], label = "", alpha = 0.4,
			markersize = 3, markerstrokewidth = 0.4)
	end

	plot(pl1, pl2, layout = (2, 1), size = (600, 500))
end

#################################################
###### Age- and sex-disaggregated analaysis #####
#################################################

function read_clean_comix2_age_sex_stratified()
	Random.seed!(123)
	df, df_part = read_comix2_df_and_df_part();
	clean_age_bins!(df_part; age_col = :part_age)
	add_sampled_ages!(df_part; age_col = :part_age, new_col = :part_age_cont)
	add_age_groups!(df_part; age_col = :part_age_cont, new_col = :part_age_grp)
	df = innerjoin(df, df_part, on = :part_id);

	@transform!(df, :cnt_age_est_bin = string.(:cnt_age_est_min, "-", :cnt_age_est_max))
	add_sampled_ages!(df; age_col = :cnt_age_est_bin, new_col = :cnt_age_cont)
	standardise_cnt_home_values!(df);
	@rename!(df, :part_id_d = :part_id)
	return (df, df_part)
end

function sample_age_from_bin(age_bin::AbstractString)
	# Parse regular bins like "18-29"
	parts = split(age_bin, "-")
	if length(parts) == 2
		lower = parse(Int, strip(parts[1]))
		upper = parse(Int, strip(parts[2]))
		return rand(lower:upper)
	end
	return parse(Int, age_bin)
end

function add_sampled_ages!(df::DataFrame; age_col = :part_age, new_col = :part_age_cont)
	@transform!(df,
		$new_col = sample_age_from_bin.(df[:, age_col]));
end

function clean_age_bins!(df::DataFrame; age_col)
	@subset!(df, @byrow !($(age_col) ∈ ["NA", "Prefer not to answer"]))
	@transform!(df,
		$age_col = replace.(
			df[:, age_col],
			"Under 1" => "0-0",
		)
	);
end

function add_age_groups!(df::DataFrame; age_col = :part_age_cont, new_col = :part_age_grp)
	breaks = [0, 18, 30, 40, 50, 60, 70, 121]
	labels = ["0-17", "18-29", "30-39", "40-49", "50-59", "60-69", "70-120"]
	df[!, new_col] = cut(df[:, age_col], breaks; labels = labels);
	nothing
end

function create_stratified_dds(df::DataFrame; key = :part_age_grp)
	df_adult_deg = degree_dist_for_all_home_non_home(df; key = key);
	gdf = groupby(df_adult_deg, key)
	dds_dic = Dict("home" => Dict(), "non-home" => Dict())
	for (k, g) in zip(keys(gdf), gdf)
		k = string(k[1])
		g_hm = @subset(g, :strat .== "home")
		g_nhm = @subset(g, :strat .== "non-home")
		dds_dic["home"][k] = DegreeDist(g_hm[:, :cnt]; include_zero = false)
		dds_dic["non-home"][k] = DegreeDist(g_nhm[:, :cnt]; include_zero = false)
	end
	return dds_dic
end

function plot_age_sex_hm_nhm_degree(df::DataFrame)
	dds_age = create_stratified_dds(df; key = :part_age_grp)
	dds_gender = create_stratified_dds(df; key = :part_gender)
	age_keys = keys(dds_age["home"]) |> collect |> sort
	gender_keys = keys(dds_gender["home"]) |> collect

	xtk = ([1, 10, 100, 1000, 10_000], [L"1", L"10", L"10^{2}", L"10^{3}", L"10^{4}"])
	kwds = (xaxis = :log10, ylim = [-5, 0.1], xlim = [1, 10_000], xticks = xtk,
		xtickfontsize = 9, ytickfontsize = 9)
	ccdf_kwds = (markersize = 1.2, markerstrokewidth = 0.0, linewidth = 0.5)

	# Age
	pl_age_hm = plot(; legendtitle = "Age", ylabel = "CCDF", title = "Home",
		left_margin = 5Plots.mm, top_margin = 2Plots.mm, kwds...)
	pl_age_nhm = plot(; legendtitle = "Age", title = "Non-home",
		legend = (0.15, 0.6), kwds...)
	for k in age_keys
		plot_ccdf!(pl_age_hm, dds_age["home"][k]; label = k, ccdf_kwds...)
		plot_ccdf!(pl_age_nhm, dds_age["non-home"][k]; label = k, ccdf_kwds...)
	end

	# Gender
	pl_gender_hm = plot(; legendtitle = "Gender",
		xlabel = "Number of contacts per day", ylabel = "CCDF", kwds...)
	pl_gender_nhm = plot(; legendtitle = "Gender",
		xlabel = "Number of contacts per day", kwds...)
	for k in gender_keys
		plot_ccdf!(pl_gender_hm, dds_gender["home"][k]; label = k, ccdf_kwds...)
		plot_ccdf!(pl_gender_nhm, dds_gender["non-home"][k]; label = k, ccdf_kwds...)
	end

	pos = (-0.1, 1.12)
	annotate!(pl_age_hm, pos, text("A", :left, 12, "Helvetica"))
	annotate!(pl_age_nhm, pos, text("B", :left, 12, "Helvetica"))
	annotate!(pl_gender_hm, pos, text("C", :left, 12, "Helvetica"))
	annotate!(pl_gender_nhm, pos, text("D", :left, 12, "Helvetica"))
	return plot(pl_age_hm, pl_age_nhm, pl_gender_hm, pl_gender_nhm,
		layout = (2, 2), dpi = 300)
end

"""
- `df_deg`: created from `degree_dist_for_all_home_non_home`
"""
function create_df_dds(df_deg::DataFrame, n_part::Int64)
	df_dds = DataFrame()
	for strat in ["all", "home", "non-home"]
		df_tmp = @pipe @subset(df_deg, :strat .== strat)[:, :cnt] |>
					   DegreeDist(_, n_part) |>
					   dd_to_df |>
					   @transform(_, :strat = strat)
		df_dds = vcat(df_dds, df_tmp)
	end
	df_dds[:, :key] .= "CoMix2 Child";
	return (df_dds)
end

function plot_child_panels(df_dds, df_ana, res_EVI)
	pl1 = plot_all_hm_nhm(df_dds, "CoMix2 Child";
		panel_name = "A", ytk_digit = 6, annotate_disp = false)
	xtk = ([1, 10, 100, 1000, 10_000], [L"1", L"10", L"10^{2}", L"10^{3}", L"10^{4}"])
	kwds = (xaxis = :log10, ylim = [-5, 0.1], xlim = [1, 10_000], xticks = xtk,
		xtickfontsize = 11, ytickfontsize = 11, legendfontsize = 10)
	plot!(pl1; legend = (0.7, 0.8),
		xlabel = "Number of contacts per day", ylabel = "CCDF", kwds...)
	pos = (-0.25, 0.98)
	annotate!(pl1, pos, text("A", :left, 17, "Helvetica"))

	df_tab = unstack(df_ana, :key, :model, :weight_waic)
	df_tab_cum = create_tab_cum(df_tab, model_names)[[2, 1], :]
	pl_bar = plot_stacked_bar(df_tab_cum, model_names;
		legend = (-0.3, -0.40),
		legend_columns = 3,
		labels = model_abbr |> values |> collect,
	)
	ytk = ([2, 1], ["Home", "Non-home"])
	plot!(pl_bar,
		left_margin = 5Plots.mm, right_margin = 0Plots.mm,
		bottom_margin = 10Plots.mm,
		yticks = ytk)

	pl_EVI = plot_EVI_across_surveys(res_EVI[[2, 1], :];
		title = "")
	plot!(pl_EVI, ytickfontsize = 10, yticks = ytk)

	pos = (-0.5, 0.98)
	annotate!(pl_bar, pos, text("B", :left, 17, "Helvetica"))
	annotate!(pl_EVI, pos, text("C", :left, 17, "Helvetica"))

	layout = @layout [a{0.6w} [b; c]]
	return plot(pl1, pl_bar, pl_EVI, layout = layout)
end

######################################
###### Setting specific analysis #####
######################################

function read_comix2_setting_strat()
	# Read cleaned data
	df_master = read_survey_master_data();
	key = "CoMix2"
	r_survey = @subset(df_master, :key .== key)[1, :];

	df, df_part = read_raw_sc_data(r_survey);
	@transform!(df_part, :country = map(x -> x[1:2], :part_id));
	df_part = filter_adult_cate(df_part; col = :part_age)
	df = innerjoin(df, df_part, on = :part_id)

	# Read main dds.
	df_dd_main, df_part = read_comix2_dds()
	n_part = nrow(df_part)

	# Prepare setting specific ones
	cnt_setting = ["cnt_work", "cnt_school", "cnt_transport", "cnt_leisure"]
	df_dd = DataFrame()
	for cnt_s in cnt_setting
		cond = df[:, cnt_s] .== true
		df_set = @pipe df[cond, :] |>
					   groupby(_, :part_id) |>
					   combine(_, nrow => :cnt)[:, :cnt] |>
					   DegreeDist(_, n_part) |> dd_to_df(_, cnt_s)
		df_dd = vcat(df_dd, df_set)
	end
	# add cnt other.
	df_tmp = @subset(df, @byrow (:cnt_home == false) & (:cnt_work == false) & (:cnt_school == false))
	df_set = @pipe df_tmp |>
				   groupby(_, :part_id) |>
				   combine(_, nrow => :cnt)[:, :cnt] |>
				   DegreeDist(_, n_part) |> dd_to_df(_, "cnt_other")
	df_dd = vcat(df_dd, df_set)

	df_nhm = @pipe @subset(df_dd_main, :strat .== "non-home") |> @select(_, Not(:key))
	df_dd = vcat(df_dd, df_nhm);
	return df_dd
end

function fit_settings(df_dd)
	lis = []
	dds = Dict(
		"work" => @subset(df_dd, :strat .== "cnt_work") |> DegreeDist,
		"school" => @subset(df_dd, :strat .== "cnt_school") |> DegreeDist,
		"transport" => @subset(df_dd, :strat .== "cnt_transport") |> DegreeDist,
		"leisure" => @subset(df_dd, :strat .== "cnt_leisure") |> DegreeDist,
		"other" => @subset(df_dd, :strat .== "cnt_other") |> DegreeDist,
	)
	cnt_setting_short = keys(dds) |> collect
	res = Dict("chns_$(s)" => Dict() for s in cnt_setting_short)
	res["dds"] = dds
	models = [model_ZeroInfNegativeBinomial, model_ZeroInfPoissonLogNormal,
		model_ZeroInfPoissonLomax]
	for k in cnt_setting_short
		Threads.@threads for model_func in models
			dd = dds[k]
			model = model_func(dd)
			med, chn = get_median_parms_from_model(model)
			chn = fit_model_with_forward_mode(model, 1000; iparms = med, progress = false)
			res["chns_$k"][get_dist_name_from_model(model_func)] = chn
		end
	end
	jldsave("../dt_intermediate/CoMix2_cnt_strat.jld2", result = res)
end

function get_df_res_setting()
	res = load("../dt_intermediate/CoMix2_cnt_strat.jld2")["result"]
	cnt_setting_short = keys(res["dds"]) |> collect

	df_sum = DataFrame()
	for strat in cnt_setting_short
		df_tmp = summarise_res_one_strat(res, strat, "CoMix2")
		df_sum = vcat(df_sum, df_tmp)
	end
	df_res = flag_minimum_IC(df_sum, :waic);
	return df_res
end

function plot_WAIC_setting(df_dd, df_res)
	df_dd_tmp = @transform(df_dd, :key = :strat)
	df_dd_tmp[:, :strat] .= "all"
	df_n_obs = @pipe groupby(df_dd_tmp, [:key]) |> combine(_, :y => sum => :n_part)
	df_n_obs[:, :strat] .= "all"
	df_res_tmp = @transform(df_res, :key = :strat);

	labels_ = ["ZInf-NB", "ZInf-PLN", "ZInf-PLomax"]
	plot_stacked_bar(df_tab_cum, model_names;
		labels = labels_,
		legend_columns = 3,
		legend = (0, -0.15), #-0.22),
		bottom_margin = 15Plots.mm,
	)
end

function prepare_block_degree_dist(df::DataFrame, df_part::DataFrame)::Dict
	block_configs = [
		(label = "Child-Child", part_cond = :<, cnt_cond = :<, key = "CoMix2 Child-Child"),
		(label = "Child-Adult", part_cond = :<, cnt_cond = :>=, key = "CoMix2 Child-Adult"),
		(label = "Adult-Child", part_cond = :>=, cnt_cond = :<, key = "CoMix2 Adult-Child"),
		(label = "Adult-Adult", part_cond = :>=, cnt_cond = :>=, key = "CoMix2 Adult-Adult"),
	]
	age_filter(col, op) = op == :>= ? (col .>= 18) : (col .< 18)

	block_results = Dict()
	for cfg in block_configs
		df_block = @subset(df,
			age_filter(:part_age_cont, cfg.part_cond),
			age_filter(:cnt_age_cont, cfg.cnt_cond)
		)
		df_block_part = @subset(df_part, age_filter(:part_age_cont, cfg.part_cond))
		n_part = df_block_part |> nrow

		df_block[:, :key] .= cfg.key
		df_block_deg = degree_dist_for_all_home_non_home(df_block; key = :key)

		# Create df_dds with proper zero-inflation using n_part
		df_dds_block = DataFrame()
		for strat in ["all", "home", "non-home"]
			df_tmp = @pipe @subset(df_block_deg, :strat .== strat)[:, :cnt] |>
						   DegreeDist(_, n_part) |>
						   dd_to_df |>
						   @transform(_, :strat = strat)
			df_dds_block = vcat(df_dds_block, df_tmp)
		end
		df_dds_block[:, :key] .= cfg.key

		block_results[cfg.label] = (
			df_block = df_block,
			df_block_part = df_block_part,
			n_part = n_part,
			df_dds = df_dds_block,
		)
		println("$(cfg.label): n_contacts=$(nrow(df_block)), n_participants=$(n_part)")
	end
	return block_results
end

function fit_block_dds_models(block_res::Dict)
	labels = keys(block_res) |> collect
	for lab in labels
		blk = block_res[lab]
		dds = Dict(
			"home" => @subset(blk.df_dds, :strat .== "home") |> DegreeDist,
			"non-home" => @subset(blk.df_dds, :strat .== "non-home") |> DegreeDist,
		)
		res = fit_hm_nhm_dds(dds)
		jldsave("../dt_intermediate/$(lab)_chns.jld2", result = res)
		println("Finished fitting: $lab → $(lab)_chns.jld2")
	end
end

function fit_block_dds_EVI(block_res::Dict)
	res_EVI_mer = DataFrame()
	labels = keys(block_res) |> collect
	for lab in labels
		blk = block_res[lab]
		dds = Dict(
			"home" => @subset(blk.df_dds, :strat .== "home") |> DegreeDist,
			"non-home" => @subset(blk.df_dds, :strat .== "non-home") |> DegreeDist,
		)
		res1 = @pipe EVI_estimate_for_qs(dds["home"], qs = [0.98]) |>
					 @transform(_, :key = "home")
		res2 = @pipe EVI_estimate_for_qs(dds["non-home"], qs = [0.98]) |>
					 @transform(_, :key = "non-home")
		res_EVI = vcat(res1, res2)
		res_EVI = @transform(res_EVI, :m_l = :mean - :lower, :m_u = :upper - :mean)
		res_EVI[!, :block] .= lab
		res_EVI_mer = vcat(res_EVI_mer, res_EVI)
	end
	return res_EVI_mer
end

function extract_alpha_values_from_block(block_res::Dict)::DataFrame
	labels = keys(block_res) |> collect
	df_alpha = DataFrame(block = String[], strat = String[], alpha = Float64[])
	for label in labels
		path = "../dt_intermediate/CoMix2_$(label)_chns.jld2"
		res = load(path)["result"]["chns_non-home"]["ZeroInfPoissonLomax"]
		d_nhome = get_ZeroInfDist(res, "ZeroInfPoissonLomax")
		push!(df_alpha, (block = label, strat = "non-home", alpha = d_nhome.d.α))
	end
	df_alpha
end

function extract_summary_stat(block_res::Dict)
	block_df_ana = DataFrame()
	labels = keys(block_res) |> collect
	for label in labels
		path = "../dt_intermediate/CoMix2_$(label)_chns.jld2"
		res = load(path)["result"]
		df_res = vcat(
			summarise_res_one_strat(res, "home", "CoMix2_$(label)"),
			summarise_res_one_strat(res, "non-home", "CoMix2_$(label)"))
		df_ana = flag_minimum_IC(df_res, :waic)
		@transform!(df_ana, :key = :strat)
		df_ana[:, :label] .= label
		block_df_ana = vcat(block_df_ana, df_ana)
		println("Loaded fitted results for $label")
	end
	return block_df_ana
end

function create_block_dds(block_res::Dict)
	block_labels = keys(block_res) |> collect
	block_dds = Dict()
	for label in block_labels
		blk = block_res[label]
		df_dds_blk = blk.df_dds
		block_dds[label] = Dict(
			"home"     => @subset(df_dds_blk, :strat .== "home") |> DegreeDist,
			"non-home" => @subset(df_dds_blk, :strat .== "non-home") |> DegreeDist,
		)
	end
	return block_dds
end

function make_waic_bar(strat, block_labels, block_df_ana; legend = true)
	block_labels = block_labels[end:-1:begin]
	ytk_bar     = (1:length(block_labels), block_labels)
	model_names = get_model_names()
	labels_abbr = model_abbr |> values |> collect

	df_tab_all = DataFrame()
	for label in block_labels
		df_strat = @subset(block_df_ana, :key .== strat, :label .== label)
		df_wide = unstack(df_strat, :key, :model, :weight_waic)
		df_wide[!, :key] .= label
		df_tab_all = vcat(df_tab_all, df_wide; cols = :union)
	end
	df_tab_cum = create_tab_cum(df_tab_all, model_names)
	pl = plot_stacked_bar(df_tab_cum, model_names;
		labels = labels_abbr, legend_columns = 3,
		legend = legend,
		right_margin = 0Plots.mm, left_margin = 5Plots.mm,
		title = "",  #strat == "home" ? "Home" : "Non-home",
		yticks = ytk_bar)
	return pl
end

function make_evi_panel(strat, block_labels, res_EVI_all, title_str)
	block_labels = block_labels[end:-1:begin]
	df_strat = @subset(res_EVI_all, :key .== strat)
	# Reorder to match block_labels order
	df_plot = DataFrame()
	for label in block_labels
		row = @subset(df_strat, :block .== label)
		df_plot = vcat(df_plot, row)
	end
	df_plot[!, :key] = df_plot[:, :block]
	pl = plot_EVI_across_surveys(df_plot; title = title_str)
	plot!(pl, ytickfontsize = 9)
	return pl
end

function plot_block_ana_panels(block_res::Dict, block_df_ana::DataFrame)
	block_labels = ["Child-Child", "Child-Adult", "Adult-Child", "Adult-Adult"]
	block_df_ana[:, :label] |> unique
	block_dds = create_block_dds(block_res)

	##### Panel A/B: CCDF overlaying 4 blocks, separated by Home / Non-home #####
	xtk = ([1, 10, 100, 1000, 10_000], [L"1", L"10", L"10^{2}", L"10^{3}", L"10^{4}"])
	ccdf_kwds = (markersize = 1.2, markerstrokewidth = 0.0, linewidth = 0.5)
	ax_kwds = (xaxis = :log10, ylim = [-5, 0.1], xlim = [1, 10_000], xticks = xtk,
		xtickfontsize = 9, ytickfontsize = 9)

	pl_hm = plot(; legendtitle = "Block", ylabel = "CCDF", title = "Home",
		left_margin = 5Plots.mm, top_margin = 2Plots.mm, ax_kwds...)
	pl_nhm = plot(; legendtitle = "Block", title = "Non-home",
		xlabel = "Number of contacts per day", ax_kwds...)

	for label in block_labels
		plot_ccdf!(pl_hm, block_dds[label]["home"]; label = label, ccdf_kwds...)
		plot_ccdf!(pl_nhm, block_dds[label]["non-home"]; label = label, ccdf_kwds...)
	end

	pl_waic_hm  = make_waic_bar("home", block_labels, block_df_ana; legend = (0.1, -0.3))
	pl_waic_nhm = make_waic_bar("non-home", block_labels, block_df_ana; legend = (0.1, -0.3))
	pl_evi_hm  = make_evi_panel("home", block_labels, res_EVI, "")
	pl_evi_nhm = make_evi_panel("non-home", block_labels, res_EVI, "")

	pos = (-0.3, 1.12)
	fontsize= 16
	annotate!(pl_hm, pos, text("A", :left, fontsize, "Helvetica"))
	annotate!(pl_nhm, pos, text("B", :left, fontsize, "Helvetica"))
	annotate!(pl_waic_hm, pos, text("C", :left, fontsize, "Helvetica"))
	annotate!(pl_waic_nhm, pos, text("D", :left, fontsize, "Helvetica"))
	annotate!(pl_evi_hm, pos, text("E", :left, fontsize, "Helvetica"))
	annotate!(pl_evi_nhm, pos, text("F", :left, fontsize, "Helvetica"))

	layout = @layout [a b; c d; e f]
	pl_block = plot(pl_hm, pl_nhm, pl_waic_hm, pl_waic_nhm, pl_evi_hm, pl_evi_nhm,
		layout = layout, size = (800, 900), dpi = 300,
		left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
	return pl_block
end
