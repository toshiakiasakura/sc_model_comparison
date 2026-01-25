
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
        ylim=[0, 105],
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
    plot!(pl2, left_margin=5Plots.mm)

    layout = @layout [a; b c{0.4w}]
    pl = plot(pl1, pl2, pl3,  layout=layout, size=(800, 600))
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
