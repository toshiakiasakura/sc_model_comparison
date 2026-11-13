
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

# Path of the bootstrap fits for sample size `M` and `rep` repeats. Lowercase
# "comix2_" is required: create_summary_stat_bootstrap matches these names with a
# case-sensitive regex, and silently skips files it cannot match.
bootstrap_path(M, rep) =
	"../dt_intermediate_bootstrap/comix2_$(M)samples_$(rep)repeat.jld2"

# Fit data bootstrapped from fitted distribution with three models.
#
# Repeats run in chunks of `chunk` rather than all at once. @memoize on
# logpdf(::ZeroInfDist) keys its cache on the distribution object and MCMC
# builds a fresh one per proposal, so the cache grows without bound; cache
# growth scales with the number of unique degrees, which grows with M. Running
# all `rep` repeats concurrently OOM-killed the process at M=10_000 (10 repeats
# needed ~31 GiB). Clearing the caches between chunks keeps peak memory flat:
# measured 19.6 GiB at M=37_347 with chunk=3.
#
# That 19.6 GiB figure understates the requirement at rep = 100: the rep=100
# run measured a 24.2 GiB peak at M=37_347 even with chunk=2 and
# --heap-size-hint=18G, leaving ~2.4 GiB free on a 27 GiB box. Hence the
# default below drops to chunk=2 at the largest M. Two further lessons from
# that run: start the largest M in a fresh process (carrying a ~10 GiB heap
# over from the previous M leaves too little headroom), and expect ~7.8 h for
# M=37_347 alone. M=30_000 is treated the same way (chunk=2).
#
# `chunk = nothing` picks per sample size; pass an integer to override.
function fit_bootstrap_data_CoMix2(M_lis; rep = 10, chunk = nothing)
	mkpath("../dt_intermediate_bootstrap")
	dists_hm, dists_nhm = get_CoMix2_fitted_dists();
	dist_LN_hm = dists_hm[2]
	dist_PLomax_nhm = dists_nhm[3]
	for M in M_lis
		# 6 fits per repeat, so chunk=2 saturates 12 threads exactly at the
		# largest M while keeping the peak well below chunk=3's.
		chunk_M = chunk === nothing ? (M >= 30_000 ? 2 : 3) : chunk
		res_mer = Vector{Any}(undef, rep)  # preallocate to avoid push! allocations
		for lo in 1:chunk_M:rep
			hi = min(lo + chunk_M - 1, rep)
			Threads.@threads for r in lo:hi
				dds = Dict(
					"home" => rand(dist_LN_hm, M) |> DegreeDist,
					"non-home" => rand(dist_PLomax_nhm, M) |> DegreeDist,
				)
				res_mer[r] = fit_hm_nhm_dds(dds)
			end
			Memoization.empty_all_caches!()
			GC.gc()
		end
		jldsave(bootstrap_path(M, rep), result = res_mer)
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

# x position on a log10 scale, with 0 placed one decade to the left of 1.
log10_pos_with_zero(k::Integer) = k == 0 ? -1.0 : log10(k)

function plot_ccdf_with_zero!(pl::Plots.Plot, d::UnivariateDistribution; kwds...)
	k = [0, (1:9)..., (10:10:90)..., (100:100:900)..., (1000:1000:9000)..., 10_000]
	ccdf.(d, k[end:-1:begin]) # To avoid the repetition errors. Create cache.
	plot!(pl, log10_pos_with_zero.(k), log10.(ccdf.(d, k)); kwds...)
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

	xtk = ([-1, 0, 1, 2, 3], [L"0", L"1", L"10", L"10^{2}", L"10^{3}"])
	ytk = ([-5, -4, -3, -2, -1, 0],
		[L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"])
	pl2 = plot(xlim = [-1, log10(5_000)], ylim = [-5, 0], ylabel = "CCDF",
		xlabel = "Number of contacts per day", xticks = xtk, yticks = ytk,
		legend = :bottomleft)
	plot_ccdf_with_zero!(pl2, d_negbin; label = "Negative binomial")
	plot_ccdf_with_zero!(pl2, d_lognormal; label = "Poisson-lognormal")
	plot_ccdf_with_zero!(pl2, d_lomax; label = "Poisson-Lomax")
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

	res = load("../dt_intermediate/CoMix2_chns.jld2")["result"]
	chns_hm = res["chns_home"]
	chns_nhm = res["chns_non-home"]
	dists_hm = [get_ZeroInfDist(chns_hm[m], m) for m in model_names]
	dists_nhm = [get_ZeroInfDist(chns_nhm[m], m) for m in model_names]
	# Panels E and F show the no-fit convolution; run `run_conv_nofit_summary()`
	# first if the cached summary is missing.
	df_grid = load(conv_nofit_summary_path())["result"]["df_grid"]

	dists_all = [get_CoMix2_all_dist(m) for m in model_names]

	default_plot_setting()
	# Best models, as used for the convolution in `fit_convoluted_dist`.
	i_best_hm, i_best_nhm = 2, 3
	xlab = "Number of contacts per day"
	kwds_font = (labelfontsize = 13, legendfontsize = 11)
	kwds_pdf = (ylim = [-5, 0], xlim = [0, 50], xlabel = xlab,
		ylabel = "Probability mass function", kwds_font...)
	kwds_ccdf = (ylim = [-5, 0], xlabel = xlab, ylabel = "CCDF", kwds_font...)
	pls = [
		plot_setting(dd_hm, dists_hm, i_best_hm, plot_pdf!; kwds_pdf...),
		plot_setting(dd_hm, dists_hm, i_best_hm, plot_ccdf!; kwds_ccdf...),
		plot_setting(dd_nhm, dists_nhm, i_best_nhm, plot_pdf!; kwds_pdf...),
		plot_setting(dd_nhm, dists_nhm, i_best_nhm, plot_ccdf!; kwds_ccdf...),
		plot_conv(dd_all, dists_all, df_grid, plot_pdf!; kwds_pdf...),
		plot_conv(dd_all, dists_all, df_grid, plot_ccdf!; kwds_ccdf...),
	]
	# Panel letters as left-aligned titles and settings as centred plot titles,
	# since annotations placed outside the axes are clipped when each panel is
	# saved on its own.
	letters = ["A", "B", "C", "D", "E", "F"]
	settings = ["Home", "Home", "Non-home", "Non-home", "All", "All"]
	fs_title = 15
	for (pl, letter, setting) in zip(pls, letters, settings)
		# A white (invisible) title reserves the space above the axes (a blank " " title
		# makes GR warn "invalid bitmap size" with Helvetica); the letter and the
		# setting are annotated on the same line there so they share size and height.
		plot!(pl, title = letter, titlefontsize = fs_title, titlefontcolor = :white)
		annotate!(pl, (0.0, 1.06), text(letter, :left, fs_title, "Helvetica"))
		annotate!(pl, (0.5, 1.06), text(setting, :center, fs_title, "Helvetica"))
		add_long_legend!(pl; x0 = 0.55, fontsize = kwds_font.legendfontsize)
	end
	return merge_panels(pls, 3, 2)
end

"""Replace the built-in legend with one drawn inside the axes, whose line
samples are long enough to show several dashes.

Note:
The GR backend fixes the legend line sample at `3 * width / 45` of the panel
(`span` in Plots' `gr.jl`), which fits only about one and a half dashes. Each
labelled series' line and marker style is read back from the panel and redrawn
as a segment of `len` (fraction of the axis width) at the top-right.

Args:
- `x0`, `y0`: left end of the first sample, as fractions of the axes.
- `dy`: vertical step between entries, as a fraction of the axes.
"""
function add_long_legend!(pl::Plots.Plot; x0 = 0.55, y0 = 0.93, dy = 0.065,
	len = 0.063, fontsize = 11)
	sp = pl[1]
	xl, yl = Plots.xlims(sp), Plots.ylims(sp)
	is_logx = sp[:xaxis][:scale] == :log10
	to_x(f) = is_logx == true ?
		10^(log10(xl[1]) + f * (log10(xl[2]) - log10(xl[1]))) :
		xl[1] + f * (xl[2] - xl[1])
	to_y(f) = yl[1] + f * (yl[2] - yl[1])

	entries = [s for s in pl.series_list if s[:label] != ""]
	plot!(pl, legend = false)
	for (i, s) in enumerate(entries)
		y = to_y(y0 - (i - 1) * dy)
		plot!(pl, [to_x(x0), to_x(x0 + len)], [y, y]; label = "",
			color = s[:linecolor], ls = s[:linestyle], lw = s[:linewidth])
		if s[:markershape] != :none
			scatter!(pl, [to_x(x0 + len / 2)], [y]; label = "",
				markershape = s[:markershape], color = s[:markercolor],
				markersize = 1.5 * s[:markersize], markerstrokewidth = 0)
		end
		annotate!(pl, to_x(x0 + len + 0.02), y,
			text(s[:label], :left, fontsize, "Helvetica"))
	end
	return pl
end

"""Save each panel as its own PNG at the default single-panel size, then tile
the images into an `nrow` x `ncol` grid.

Note:
Rendering the panels as one multi-panel `plot` shrinks the legend line samples
until the dashed styles are hard to tell apart, so each panel is drawn on its
own and only the images are merged. Returns an image matrix; save it with
`Images.save`.
"""
function merge_panels(pls::Vector, nrow::Int64, ncol::Int64; dpi = 300,
	margin = 5Plots.mm)
	dir = mktempdir()
	imgs = map(enumerate(pls)) do (i, pl)
		path = joinpath(dir, "panel_$i.png")
		savefig(plot(pl; dpi = dpi, margin = margin), path)
		load(path)
	end
	rows = [hcat(imgs[(r-1)*ncol+1:r*ncol]...) for r in 1:nrow]
	return vcat(rows...)
end

"""One setting (home or non-home): the observed degree distribution and the
three fitted zero-inflated Poisson mixtures, with "(best)" marking the best model.
"""
function plot_setting(dd::DegreeDist, dists, i_best::Int64, plot_func!; kwds...)
	pl = plot(; kwds...)
	colors = [7, 6, 13]
	kwds_obs = (markersize = 2.5, markerstrokewidth = 0.2)
	kwds_fit = (lw = 1.5, ls = :dash)
	plot_func!(pl, dd, color = :black, label = "Observed"; kwds_obs...)
	for (i, (m, dist)) in enumerate(zip(get_model_names(), dists))
		label = model_abbr[m] * (i == i_best ? " (best)" : "")
		plot_func!(pl, dist, color = colors[i], label = label; kwds_fit...)
	end
	return pl
end

"""
Args:
- `df_grid`: `out["df_grid"]` from `summarise_conv_nofit`. The convoluted curve
  is the no-fit one (each posterior draw of the home parameters convoluted with
  a draw of the non-home parameters, then the pointwise median), not the
  refitted `model_ZeroInfConvDist`.
"""
function plot_conv(dd_all, dists_all, df_grid::DataFrame,
	plot_func!; kwds...)
	colors = [7, 6, 13]
	kwds_obs = (markersize = 2.5, markerstrokewidth = 0.0)
	kwds_fit = (lw = 1.5, ls = :dash)

	pl = plot(; kwds...)
	plot_func!(pl, dd_all, color = :black, label = "Observed"; kwds_obs...)
	for (i, (m, dist)) in enumerate(zip(get_model_names(), dists_all))
		plot_func!(pl, dist, color = colors[i], label = model_abbr[m]; kwds_fit...)
	end
	# Solid, so it is not confused with the dashed ZInf-NB curve
	plot_conv_nofit_curve!(pl, df_grid, plot_func!; band = false,
		color = colorant"#E69F00", ls = :solid, lw = 2, label = "Convolution")
	return pl
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

######################################################
##### Convolution without refitting ##################
######################################################

conv_nofit_summary_path() = "../dt_intermediate/CoMix2_conv_nofit_summary.jld2"

"""The k grid used by `plot_pdf!`/`plot_ccdf!` in `distributions/plot.jl`.

Note:
Returned by a function rather than stored in a `const`, because the notebook
cells re-`include` this file and redefining a `const` array warns.
"""
function conv_k_grid(; include_zero = true)
	k = [(1:9)..., (10:10:90)..., (100:100:900)..., (1000:1000:9000)..., 10_000]
	return include_zero == true ? [0, k...] : k
end

# log10 that drops non-positive values instead of returning -Inf, which would
# otherwise stretch a filled band down to the axis floor.
safe_log10(y::Real) = y > 0 ? log10(y) : NaN

"""Probability mass vector of a `ZeroInfDist` on `0:K`; `pmf[k+1]` is P(X=k).

Note:
- `logpdf(::PoissonMixture, k)` is called deliberately. `pdf(::PoissonMixture, k)`
  and `logpdf(::ZeroInfDist, k)` are `@memoize`d, and caching K+1 values for
  every posterior draw would grow the never-evicted caches to ~10^7 entries.
- `quadgk` can return a tiny negative or a non-finite value at extreme k; both
  are flattened to 0 so that the reverse cumsum stays monotone.
"""
function zeroinf_pmf_vector(d::ZeroInfDist, K::Int64)::Vector{Float64}
	pmf = exp.(logpdf.(Ref(d.d), 0:K))
	pmf = [isfinite(p) && p > 0.0 ? p : 0.0 for p in pmf]
	pmf .*= (1 - d.π0)
	pmf[1] += d.π0
	return pmf
end

"""Convolve a home and a non-home probability mass vector.

Args:
- `pmf_hm`, `pmf_nhm`: probability mass vectors indexed as `0:K`.
- `k_h_max`: maximum home degree. Only the first `k_h_max + 1` home entries are
  used, reproducing exactly the truncation of
  `pdf(::ZeroInfConvolutedDist, k)` in `distributions/zeroinf.jl`.
"""
function convolute_pmf_vectors(pmf_hm::Vector{Float64}, pmf_nhm::Vector{Float64},
	k_h_max::Int64)::Vector{Float64}
	K = length(pmf_nhm) - 1
	i_max = minimum((k_h_max, K, length(pmf_hm) - 1))
	pmf = zeros(Float64, K + 1)
	for i in 0:i_max
		p_h = pmf_hm[i+1]
		if p_h == 0.0
			continue
		end
		@inbounds @simd for k in i:K
			pmf[k+1] += p_h * pmf_nhm[k-i+1]
		end
	end
	return pmf
end

"""Convoluted ("all contacts") probability mass vector on `0:K` for one
posterior draw.

Note:
`K = 10_000` matches `ccdf(::ZeroInfConvolutedDist, k; k_max = 10_000)`, so the
no-fit and the refitted curves are truncated identically.
"""
function conv_pmf_one_draw(d_hm::ZeroInfDist, d_nhm::ZeroInfDist, k_h_max::Int64;
	K::Int64 = 10_000)::Vector{Float64}
	pmf_hm = zeroinf_pmf_vector(d_hm, k_h_max)
	pmf_nhm = zeroinf_pmf_vector(d_nhm, K)
	return convolute_pmf_vectors(pmf_hm, pmf_nhm, k_h_max)
end

# Same, for an existing ZeroInfConvolutedDist (e.g. the refitted median-parameter
# distribution), so that both routes go through identical arithmetic.
conv_pmf_one_draw(conv_d::ZeroInfConvolutedDist; K::Int64 = 10_000) =
	conv_pmf_one_draw(conv_d.d_hm, conv_d.d_nhm, conv_d.k_h_max; K = K)

"""Zero-truncated CCDF of a convoluted probability mass vector, on `0:K`.

Note:
`obtain_ccdf` returns P(X >= k), matching `ccdf(::ZeroInfConvolutedDist, k)`,
and dividing by `1 - pmf[1]` matches `ZeroTruncConvolutedDist`, which is what
both `plot_ccdf!(pl, ::ZeroInfConvolutedDist)` and `plot_ccdf!(pl, ::DegreeDist)`
(zero class dropped) display.
"""
zerotrunc_ccdf_from_pmf(pmf::Vector{Float64}) = obtain_ccdf(pmf) ./ (1 - pmf[1])

"""One posterior draw: the convoluted pmf and zero-truncated CCDF restricted to
the plot grid `ks`, the analytic mean, and two truncation diagnostics.
"""
function conv_nofit_one_draw(d_hm::ZeroInfDist, d_nhm::ZeroInfDist,
	k_h_max::Int64, ks::Vector{Int64}; K::Int64 = 10_000)
	pmf = conv_pmf_one_draw(d_hm, d_nhm, k_h_max; K = K)
	ccdf_ = zerotrunc_ccdf_from_pmf(pmf)
	return (
		pmf = pmf[ks .+ 1],
		ccdf = ccdf_[ks .+ 1],
		mean = mean(d_hm) + mean(d_nhm), # Independence: the means add.
		mean_trunc = sum((0:K) .* pmf),  # Diagnostic only, biased low.
		mass = sum(pmf),                 # < 1 by the truncated Lomax tail.
	)
end

"""Posterior quantiles of the three values of a column set, ignoring
non-finite draws.
"""
function quantile_over_draws(M::Matrix{Float64}, qs = [0.025, 0.5, 0.975])
	out = Vector{Vector{Float64}}(undef, size(M, 2))
	for c in 1:size(M, 2)
		v = filter(isfinite, M[:, c])
		out[c] = length(v) == 0 ? fill(NaN, length(qs)) : quantile(v, qs)
	end
	return out
end

"""Posterior of the convoluted ("all contacts") degree distribution *without*
refitting.

Every posterior draw of the home parameters is paired with a draw of the
non-home parameters and the two zero-inflated pmfs are convoluted, giving a
posterior set of "all contacts" distributions. The point estimate is the
pointwise median across that set.

Args:
- `chn_hm`, `chn_nhm`: marginal chains, e.g. `res["chns_home"][model_names[2]]`.
- `dd_hm`: home degree distribution; `maximum(dd_hm)` gives `k_h_max`.
- `key_hm`, `key_nhm`: model names required by `get_vec_ZeroInfDist_from_chn`.
- `K`: pmf truncation; 10_000 matches `ccdf(::ZeroInfConvolutedDist)`.
- `thin`: use every `thin`-th draw (`thin = 50` for a quick smoke test).
- `threaded`: safe as written, because the inner loop only touches the
  non-memoized `logpdf(::PoissonMixture, ::Int64)`. Never reintroduce
  `pdf(::PoissonMixture, k)` or `pdf(::ZeroInfConvolutedDist, k)` there:
  Memoization.jl writes into a shared, never-evicted Dict.

Returns a `Dict` suitable for `jldsave`.
"""
function summarise_conv_nofit(chn_hm::Chains, chn_nhm::Chains, dd_hm::DegreeDist;
	key_hm::String = get_model_names()[2],
	key_nhm::String = get_model_names()[3],
	K::Int64 = 10_000, thin::Int64 = 1, threaded::Bool = true)::Dict

	for (lbl, chn) in (("home", chn_hm), ("non-home", chn_nhm))
		if size(chn, 3) != 1
			println("$(lbl) chain holds $(size(chn, 3)) chains; ",
				"get_vec_ZeroInfDist_from_chn reads chain 1 only.")
		end
	end
	dists_hm = get_vec_ZeroInfDist_from_chn(chn_hm, key_hm)
	dists_nhm = get_vec_ZeroInfDist_from_chn(chn_nhm, key_nhm)
	n_draw = minimum((length(dists_hm), length(dists_nhm)))
	if length(dists_hm) != length(dists_nhm)
		println("Chain lengths differ ($(length(dists_hm)) vs ",
			"$(length(dists_nhm))): the first $(n_draw) draws are paired.")
	end

	# The two chains were fitted independently, so pairing draws index-wise is a
	# legitimate sample from the product posterior, and it is reproducible.
	inds = collect(1:thin:n_draw)
	k_h_max = maximum(dd_hm)
	ks = conv_k_grid(; include_zero = true)
	n_s = length(inds)
	rows = Vector{Any}(undef, n_s) # Preallocated: required by Threads.@threads.

	if threaded == true
		Threads.@threads for j in 1:n_s
			rows[j] = conv_nofit_one_draw(dists_hm[inds[j]], dists_nhm[inds[j]],
				k_h_max, ks; K = K)
		end
	else
		for j in 1:n_s
			rows[j] = conv_nofit_one_draw(dists_hm[inds[j]], dists_nhm[inds[j]],
				k_h_max, ks; K = K)
		end
	end

	mean_set = [r.mean for r in rows]
	mean_trunc_set = [r.mean_trunc for r in rows]
	mass_set = [r.mass for r in rows]

	df_grid = DataFrame(:k => ks)
	for (nm, M) in (("pmf", reduce(vcat, [r.pmf' for r in rows])),
		("ccdf", reduce(vcat, [r.ccdf' for r in rows])))
		qs = quantile_over_draws(M)
		df_grid[!, Symbol("$(nm)_l")] = getindex.(qs, 1)
		df_grid[!, Symbol("$(nm)_med")] = getindex.(qs, 2)
		df_grid[!, Symbol("$(nm)_u")] = getindex.(qs, 3)
	end

	# mean(::ZeroInfDist) is NaN when a PoissonLomax draw has α <= 1, so NaN
	# draws are dropped as in `summarise_res_one_strat`.
	cond = isnan.(mean_set)
	if all(cond) == true
		println("All NaN: no draw has a finite mean.")
		mean_med, mean_l, mean_u = NaN, NaN, NaN
	else
		if any(cond) == true
			println("NaN presence in the posterior means: ",
				"$(sum(cond))/$(n_s) draws (PoissonLomax α <= 1)")
		end
		mean_med = median(mean_set[.!cond])
		mean_l, mean_u = quantile(mean_set[.!cond], [0.025, 0.975])
	end
	df_mean = DataFrame(tp = "Convoluted (no-fit)",
		mean = mean_med, mean_l = mean_l, mean_u = mean_u,
		mean_trunc = median(mean_trunc_set),
		pc_nan = sum(cond) / n_s * 100,
		mass_med = median(mass_set), mass_min = minimum(mass_set),
		n_draw = n_s)

	return Dict(
		"df_grid" => df_grid, "df_mean" => df_mean,
		"mean_set" => mean_set, "mean_trunc_set" => mean_trunc_set,
		"mass_set" => mass_set,
		"K" => K, "k_h_max" => k_h_max, "thin" => thin,
		"key_hm" => key_hm, "key_nhm" => key_nhm)
end

"""Load the marginal CoMix2 chains, build the no-fit convolution posterior and
cache the summary.
"""
function run_conv_nofit_summary(; K::Int64 = 10_000, thin::Int64 = 1,
	threaded::Bool = true, save::Bool = true)::Dict
	dd_all, dd_hm, dd_nhm = get_comix2_dd_all_hm_nhm()
	model_names = get_model_names()
	res = load("../dt_intermediate/CoMix2_chns.jld2")["result"]
	chn_hm = res["chns_home"][model_names[2]]
	chn_nhm = res["chns_non-home"][model_names[3]]

	out = summarise_conv_nofit(chn_hm, chn_nhm, dd_hm;
		key_hm = model_names[2], key_nhm = model_names[3],
		K = K, thin = thin, threaded = threaded)
	out["mean_obs_all"] = mean(dd_all)
	out["mean_obs_hm_plus_nhm"] = mean(dd_hm) + mean(dd_nhm)
	if save == true
		jldsave(conv_nofit_summary_path(); result = out)
		println("Saved: ", conv_nofit_summary_path())
	end
	return out
end

"""Best 2-parameter marginal fit to the "all contacts" degree distribution.

Note:
Two files coexist. The newer `CoMix2_all_chns.jld2` (key `res`, written by
`fit_CoMix2_all`) holds all three chains; the older `CoMix2_chns_all.jld2`
(key `result`) holds a bare `Chains` for one model and is what `plot_conv_fit`
reads.
"""
function get_CoMix2_all_dist(model_name::String = get_model_names()[3])
	path_new = "../dt_intermediate/CoMix2_all_chns.jld2"
	if isfile(path_new) == true
		chn = load(path_new)["res"]["chns_all"][model_name]
	else
		chn = load("../dt_intermediate/CoMix2_chns_all.jld2")["result"]
	end
	return get_ZeroInfDist(chn, model_name)
end

"""Add the no-fit convolution posterior to an existing panel: the pointwise
median, and optionally the 95% credible band.

Args:
- `df_grid`: `out["df_grid"]` from `summarise_conv_nofit`.
- `plot_func!`: `plot_pdf!` or `plot_ccdf!`. Used only to pick the pmf or the
  CCDF columns and the k range, not to draw.

Note:
The pmf panel uses the zero-inflated pmf, because `plot_pdf!(::DegreeDist)`
normalises including the zero class, while the CCDF panel uses the
zero-truncated CCDF and skips k = 0, because `plot_ccdf!(::DegreeDist)` drops
the zero class and the x axis is on a log10 scale.
"""
function plot_conv_nofit_curve!(pl::Plots.Plot, df_grid::DataFrame, plot_func!;
	band = true, color = :red, ls = :dash, lw = 1.5,
	label = "Convoluted (no-fit)", label_band = "", kwds...)
	is_ccdf = nameof(plot_func!) == :plot_ccdf!
	col = is_ccdf == true ? "ccdf" : "pmf"
	df = is_ccdf == true ? @subset(df_grid, :k .>= 1) : df_grid

	k = df[:, :k]
	if band == true
		plot!(pl, k, safe_log10.(df[:, Symbol("$(col)_l")]);
			fillrange = safe_log10.(df[:, Symbol("$(col)_u")]),
			fillalpha = 0.3, lw = 0, color = :gray, label = label_band)
	end
	plot!(pl, k, safe_log10.(df[:, Symbol("$(col)_med")]);
		color = color, lw = lw, ls = ls, label = label, kwds...)
	return pl
end

"""One panel overlaying the no-fit convolution posterior on the observed
"all contacts" degree distribution.

Args:
- `df_grid`: `out["df_grid"]` from `summarise_conv_nofit`.
- `plot_func!`: `plot_pdf!` or `plot_ccdf!`, as in `plot_conv`.

Note:
The pmf panel uses the zero-inflated pmf, because `plot_pdf!(::DegreeDist)`
normalises including the zero class, while the CCDF panel uses the
zero-truncated CCDF and skips k = 0, because `plot_ccdf!(::DegreeDist)` drops
the zero class and the x axis is on a log10 scale.
"""
function plot_conv_nofit(dd_all::DegreeDist, df_grid::DataFrame,
	dist_all, conv_dist::ZeroInfConvolutedDist, plot_func!; kwds...)
	kwds_obs = (markersize = 2.5, markerstrokewidth = 0.0)
	kwds_fit = (lw = 1.5, ls = :dash)

	pl = plot(; kwds...)
	plot_func!(pl, dd_all, label = "All (observed)"; kwds_obs...)
	plot_func!(pl, dist_all, label = "Best 2-parameter dist"; color = 13, kwds_fit...)
	# Drawn before the refitted curve, and in black rather than the observed
	# data's blue, so that the two convolutions remain distinguishable where
	# they coincide.
	plot_conv_nofit_curve!(pl, df_grid, plot_func!; band = true,
		color = :black, ls = :solid, lw = 2.0,
		label = "Convoluted (no-fit)",
		label_band = "Convoluted (no-fit), 95% CrI")
	plot_func!(pl, conv_dist, label = "Convoluted (refitted)"; color = :red, kwds_fit...)
	return pl
end

"""Two-panel figure (pmf and CCDF) for the no-fit convolution.

Args:
- `out`: returned by `run_conv_nofit_summary`, or loaded from
  `conv_nofit_summary_path()`.
"""
function plot_conv_nofit_panels(out::Dict)
	dd_all, dd_hm, dd_nhm = get_comix2_dd_all_hm_nhm()
	model_names = get_model_names()
	chn_conv = load("../dt_intermediate/CoMix2_convoluted_chns.jld2")["result"]
	conv_dist = get_ZeroInfConvDist(chn_conv, dd_hm)
	dist_all = get_CoMix2_all_dist(model_names[3])

	xlab = "Number of contacts per day"
	pl1 = plot_conv_nofit(dd_all, out["df_grid"], dist_all, conv_dist, plot_pdf!;
		ylim = [-5, 0], xlim = [0, 50])
	pl2 = plot_conv_nofit(dd_all, out["df_grid"], dist_all, conv_dist, plot_ccdf!;
		ylim = [-5, 0], legend = :bottomleft)
	plot!(pl1, xlabel = xlab, ylabel = "Probability mass function",
		left_margin = 5Plots.mm)
	plot!(pl2, xlabel = xlab, ylabel = "CCDF")

	pos = (-0.2, 1.0)
	annotate!(pl1, pos, text("A", :left, 18, "Helvetica"))
	annotate!(pl2, pos, text("B", :left, 18, "Helvetica"))
	return plot(pl1, pl2, layout = (1, 2), size = (900, 400), dpi = 300,
		fig = :png, bottom_margin = 5Plots.mm, top_margin = 5Plots.mm)
end

"""Numeric comparison of the means: observed, no-fit convolution, refitted
convolution and the best 2-parameter fit to the "all contacts" data.

Note:
There is no `mean(::ZeroInfConvolutedDist)` method, so the component means of
the refitted convolution are added explicitly.
"""
function compare_conv_nofit_means(out::Dict)::DataFrame
	dd_all, dd_hm, dd_nhm = get_comix2_dd_all_hm_nhm()
	chn_conv = load("../dt_intermediate/CoMix2_convoluted_chns.jld2")["result"]
	conv_dist = get_ZeroInfConvDist(chn_conv, dd_hm)
	dist_all = get_CoMix2_all_dist()
	r = out["df_mean"][1, :]

	println("Truncation at K = $(out["K"]): median pmf mass = ",
		r.mass_med, ", minimum = ", r.mass_min)
	println("Median truncated mean (diagnostic, biased low) = ", r.mean_trunc)
	println("Draws with a non-finite mean: ", r.pc_nan, "%")

	df = DataFrame(
		tp = ["All (observed)", "Home + non-home (observed)",
			"Convoluted (no-fit), median", "Convoluted (refitted), median parms",
			"Best 2-parameter dist"],
		mean = [mean(dd_all), mean(dd_hm) + mean(dd_nhm), r.mean,
			mean(conv_dist.d_hm) + mean(conv_dist.d_nhm), mean(dist_all)],
		mean_l = [NaN, NaN, r.mean_l, NaN, NaN],
		mean_u = [NaN, NaN, r.mean_u, NaN, NaN])
	return df
end

###############################
##### Sample size effect ######
###############################

"""
Note:
See `create_summary_stat_one_data` for a similar function.
"""
function create_summary_stat_bootstrap(; sample_sizes = nothing)
	paths = glob("../dt_intermediate_bootstrap/*.jld2")

	# Keep one file per sample size: the one with the most repeats. Older
	# low-repeat runs are left on disk, and processing both would emit two rows
	# for the same (sample_size, iteration), which flag_minimum_IC then joins
	# into a cross product.
	best = Dict{Int, Tuple{Int, String}}()
	for path in paths
		m = match(r"comix2_(\d+)samples_(\d+)repeat", path)
		if m === nothing
			continue
		end
		sample_size = parse(Int, m.captures[1])
		n_repeat = parse(Int, m.captures[2])
		if !haskey(best, sample_size) || n_repeat > best[sample_size][1]
			best[sample_size] = (n_repeat, path)
		end
	end

	df_all = DataFrame()
	# `sample_sizes` restricts the summary to those sample sizes, e.g. to skip
	# M=37_347 files left on disk from earlier runs.
	ss_use = isnothing(sample_sizes) ? sort(collect(keys(best))) : sort(sample_sizes)
	for sample_size in ss_use
		haskey(best, sample_size) || error("No bootstrap file for M=$(sample_size)")
		n_repeat, path = best[sample_size]
		df_all = vcat(df_all, summarise_bootstrap_file(path, sample_size))
	end
	return df_all
end

"""
Summarise one bootstrap file (`bootstrap_path(M, rep)`). `iters` selects a
subset of repeats so that a large file can be split across processes.

The memoized `logpdf` caches are keyed on distribution objects, which are
rebuilt for every repeat, so they are never reused across repeats. Clearing them
after each repeat keeps memory flat without changing the results; without it a
100-repeat summary grew past 20 GiB.
"""
function summarise_bootstrap_file(path, sample_size; iters = nothing)
	res_mer = load(path)["result"]
	iters = isnothing(iters) ? eachindex(res_mer) : iters
	df_all = DataFrame()
	for i in iters
		for strat in ["home", "non-home"]
			df_tmp = summarise_res_one_strat(res_mer[i], strat, "CoMix2_bootstrap")
			df_tmp[!, :sample_size] .= sample_size
			df_tmp[!, :iteration] .= i
			df_all = vcat(df_all, df_tmp)
		end
		Memoization.empty_all_caches!()
		println("M=$(sample_size): repeat $(i) done"); flush(stdout)
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

	# Ticks at the simulated sample sizes, with thousands separators.
	ss = sort(unique(df_boot[:, :n_sample]))
	fmt_ss(n) = replace(string(Int(n)), r"(?<=\d)(?=(\d{3})+$)" => ",")
	pl2 = plot(xlabel = "sample size", ylabel = "WAIC weight",
		xticks = (log10.(ss), fmt_ss.(ss)), xlim = xlims(pl1), legend = nothing)
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

# Path of the age-block fits. Shared by the writer and readers below: these had
# drifted, with the readers looking for a "CoMix2_" prefix the writer never
# produced, which left the age-block cells unable to load their own output.
block_chns_path(label) = "../dt_intermediate/$(label)_chns.jld2"

function fit_block_dds_models(block_res::Dict)
	labels = keys(block_res) |> collect
	for lab in labels
		blk = block_res[lab]
		dds = Dict(
			"home" => @subset(blk.df_dds, :strat .== "home") |> DegreeDist,
			"non-home" => @subset(blk.df_dds, :strat .== "non-home") |> DegreeDist,
		)
		res = fit_hm_nhm_dds(dds)
		jldsave(block_chns_path(lab), result = res)
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
		path = block_chns_path(label)
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
		path = block_chns_path(label)
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
