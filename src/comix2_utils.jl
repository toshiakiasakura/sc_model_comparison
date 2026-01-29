
###############################
##### Simulation analysis #####
###############################

function get_ZInf_PLomax_mean()
    res = load("../dt_intermediate/CoMix2_chns.jld2")["result"];
    chn = res["chns_non-home"][model_names[3]]
    d_lomax = get_ZeroInfDist(chn, model_names[3]);
    return mean(d_lomax)
end

function get_ZInf_means()
    res = load("../dt_intermediate/CoMix2_chns.jld2")["result"];
    ms = []
    for m in model_names
        chn = res["chns_non-home"][m]
        d = get_ZeroInfDist(chn, m);
        push!(ms, mean(d))
    end
    return ms
end

"""
Args:
- df_sum: returned by `parse_bootstrap_estimated_data`.
"""
function plot_coverage_prob(df_sum::DataFrame)
    ms = ["ZInf-NB", "ZInf-PLN", "ZInf-PLomax"]
    rep_dic = Dict( m => m_new for (m, m_new) in zip(model_names, ms))
    df_cov = @pipe groupby(df_sum, [:tp, :sample_size]) |> combine(_) do gdf
        n  = nrow(gdf)
        n_cov = sum(gdf[:, :cov_flag])
        (cov_prob=n_cov/n*100, )
    end |> @subset(_, :tp .!= "Sample mean") |>
        @transform(_, :tp = replace.(:tp, rep_dic...),
                    :sample_size = string.(:sample_size))
    pl = plot(; xlabel="Number of samples per simulation",
        ylabel="Coverage probability (%)"
    )
    plot!(pl,
        df_cov[:, :sample_size], df_cov[:, :cov_prob], group=df_cov[:, :tp],
        marker=:circle, markerstrokewidth = 0.5, color= [7 6 13],
        legend = (0.7, 0.5),
    )
    return pl
end

"""
Args:
- df_mer: returned by `simulate_and_take_sample_mean`.
"""
function plot_simulated_sample_mean(df_mer::DataFrame; color=1)
    pls = []
    tps = ["Bootstrap", "ZInf-NB", "ZInf-PLN", "ZInf-PLomax"]
    ms = get_ZInf_means()
    ms = vcat([mean(dd_nhm)], ms)

    for (i, tp) in enumerate(tps)
        df_tmp = @subset(df_mer, :tp .== tp)
        ylbl = i == 1 ? "Sample mean of simulated data" : ""
        m = ms[i]
        ytk = i == 1 ? true : false

        pl = plot(; ylim=[0, 10], xrotation=30)
        dotplot!(pl, df_tmp[:, :sample_size], df_tmp[:, :mean],
            ylabel=ylbl, xlabel="", yticks = ytk,
            legend=false, color=color,
            markerstrokewidth = 0.1, markersize=2.2)
        annotate!(pl, (0.5, 0.1,
        ), text(tp, :black, :centre, 10, "Helvetica"))
        boxplot!(pl, df_tmp[:, :sample_size], df_tmp[:, :mean],
            color=color, outliers=false, fillalpha=0.75)
        hline!(pl, [m], ls= :dot, color = 2, lw=3)
        push!(pls, pl)
    end
    plot!(pls[4], right_margin = 5Plots.mm, bottom_margin=10Plots.mm)
    annotate!(pls[2], (1.0, -0.22),
        text("Number of samples per simulation",
            :black, :centre, 12, "Helvetica"))
    return plot(pls..., layout=(1, 4), size= (800, 400), )
end

function parse_bootstrap_estimated_data(paths)
    m_PLomax = get_ZInf_PLomax_mean()

    model_names = get_model_names()
    df_sum = DataFrame()
    for (i, path) in enumerate(paths)
        M = M_lis[i]
        res = load(path)["result"];
        N = length(res)
        for m in model_names
            chns = [res[i]["chns_non-home"][m] for i in 1:N]
            ms = [get_ZeroInfDist(chns[i], m) |> mean for i in 1:N]
            cov_flag = [
                @pipe get_vec_ZeroInfDist_from_chn(chns[i], m) .|> mean |>
                (x -> x[.~isnan.(x)]) |>
                quantile(_, [0.025, 0.975]) |> (x -> (x[1] < m_PLomax < x[2]))
                for i in 1:N
            ]
            df_m = DataFrame(mean = ms, sample_size = M, tp = m, cov_flag=cov_flag)
            df_sum = vcat(df_sum, df_m)
        end
        ms = [ res[i]["dds"]["non-home"] |> mean for i in 1:N]
        df_m = DataFrame(mean = ms, sample_size = M, tp = "Sample mean", cov_flag=false)
        df_sum = vcat(df_sum, df_m)
    end
    df_sum
end

"""
Args:
- df_sum: returned by `parse_bootstrap_estimated_data`,
"""
function plot_estimated_means(df_sum::DataFrame)
    m_PLomax = get_ZInf_PLomax_mean()
    order = ["Sample mean", "ZInf-NB", "ZInf-PLN", "ZInf-Lomax"]
    rep_dic = Dict( m => m_new for (m, m_new) in zip(["Sample mean", model_names...], order) )
    @transform!(df_sum, :tp = replace(:tp, rep_dic...));
    # Check NaN
    cond = isnan.(df_sum[:, :mean])
    df_sum[cond, :] #|> display
    df_sum_vis = df_sum[.~cond, :];
    df_sum_vis[!, :sample_size] = string.(df_sum_vis[:, :sample_size])
    df_sum_vis = @subset(df_sum_vis, :tp .!= "Sample mean")
    pl = plot_mean(df_sum_vis, m_PLomax;
        order = order,
        ylabel="Estimated means",
        color = [7 6 13],
        title="")
    return pl
end

function simulate_and_take_sample_mean()
    res = load("../dt_intermediate/CoMix2_chns.jld2")["result"];
    model_names = get_model_names();
    dd_all, dd_hm, dd_nhm = get_comix2_dd_all_hm_nhm()

    dists = []
    for m in model_names
        chn = res["chns_non-home"][m]
        push!(dists, get_ZeroInfDist(chn, m))
    end

    N = 100
    df_mer = DataFrame()
    M_lis = [100, 1000, 10000, 37347]
    for M in M_lis
        boot_sim = [rand(dd_nhm, M) |> mean for _ in 1:N]
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

function plot_mean(df_vis::DataFrame, h_m::Real;
	order = ["ZInf-NB", "ZInf-LN", "ZInf-Lomax"],
    ylabel = "Sample mean of simulated data",
    color = [1 7 6 13],
	kwds...)
	df_vis = copy(df_vis)
	#order = ["Observed", "ZeroTruncNegativeBinomial", "ZeroTruncPoissonLomax", "Best"]
	df_vis.tp = categorical(df_vis.tp, levels = order, ordered = true)

	ylim = [0, 10.0]
	pl = plot(ylim = ylim, ylabel = ylabel, xlabel = "Number of samples per simulation"; kwds...)
	groupeddotplot!(pl, df_vis[:, :sample_size], df_vis[:, :mean];
		group = df_vis[:, :tp], label = "", color = color,
		markersize = 2.2, markerstrokewidth = 0.2)
	groupedboxplot!(pl, df_vis[:, :sample_size], df_vis[:, :mean]; group = df_vis[:, :tp],
		fillalpha = 0.75, outliers = false, color = color)
	hline!([h_m], ls = :dash, color = :black, label = "") # , label="Baseline eigenvalue")
	return pl
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
    chn = sample(model, NUTS(), 1000; progress = true)
    jldsave("../dt_intermediate/CoMix2_convoluted_chns.jld2", result=chn)

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

    chn = load("../dt_intermediate/CoMix2_chns_all.jld2")["result"]
    dist_all = get_ZeroInfDist(chn, model_names[3])

    pl1 = plot_hm_nhm(dd_hm, dd_nhm, dists_hm, dists_nhm, plot_pdf!; ylim=[-5, 0], xlim=[0,50])
    pl2 = plot_hm_nhm(dd_hm, dd_nhm, dists_hm, dists_nhm, plot_ccdf!; ylim=[-5, 0])
    xlab = "Number of contacts per day"
    plot!(pl1, xlabel=xlab, ylabel="Probability density function",
        left_margin=5Plots.mm)
    plot!(pl2, xlabel=xlab, ylabel="CCDF")
    plot(pl1, pl2, size =(800, 400), bottom_margin=5Plots.mm)

    pl3 = plot_conv(dd_all, dist_all, conv_dist, plot_pdf!; ylim=[-5, 0], xlim=[0,50])
    pl4 = plot_conv(dd_all, dist_all, conv_dist, plot_ccdf!; ylim=[-5, 0])
    plot!(pl3, xlabel=xlab, ylabel="Probability density function",
        left_margin=5Plots.mm
    )
    plot!(pl4, xlabel=xlab, ylabel="CCDF")

    layout = @layout [a b; c d]
    return plot(pl1, pl2, pl3, pl4, layout=layout, size=(800, 800), dpi=150, fig=:png)
end

function plot_hm_nhm(dd_hm::DegreeDist, dd_nhm::DegreeDist, dists_hm, dists_nhm, plot_func!;
		kwds...
	)
    pl = plot(; kwds...)
    colors = [7, 6, 13]
    color = 2
	kwds_obs = (markersize=2.5, markerstrokewidth=0.2)
	kwds_fit = (lw=1.5, ls=:dash)
    plot_func!(pl, dd_hm, color=color, label="Home (observed)"; kwds_obs...)
    plot_func!(pl, dists_hm[2], color=color, ls=:dash, label="Home, Best fitted"; kwds_fit...)
    color = 1
    plot_func!(pl, dd_nhm, color=color, label="Non-home (observed)"; kwds_obs...)
    plot_func!(pl, dists_nhm[1], color=colors[1], label="Non-home, ZInf-NB"; kwds_fit...)
    plot_func!(pl, dists_nhm[2], color=colors[2], label="Non-home, ZInf-PLN"; kwds_fit...)
    plot_func!(pl, dists_nhm[3], color=colors[3], label="Non-home, ZInf-PLomax"; kwds_fit...)
	return pl
end

function plot_conv(dd_all, dist_all, conv_dist, plot_func!; kwds...)
	kwds_obs = (markersize=2.5, markerstrokewidth=0.0)
	kwds_fit = (lw=1.5, ls=:dash)

    pl = plot(; kwds...)
    plot_func!(pl, dd_all, label="All (observed)"; kwds_obs...)
    plot_func!(pl, dist_all, label="Best 2-parameter dist"; color=13, kwds_fit...)
    plot_func!(pl, conv_dist, label="Convoluted"; color = :red, kwds_fit...)
    #plot_func!(pl, conv_dist_refit, label="Convoluted (no-fit)"; kwds_fit...)
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
    df_dd_tmp =  @transform(df_dd, :key = :strat)
    df_dd_tmp[:, :strat] .= "all"
    df_n_obs = @pipe groupby(df_dd_tmp, [:key]) |> combine(_, :y => sum => :n_part)
    df_n_obs[:, :strat] .= "all"
    df_res_tmp = @transform(df_res, :key = :strat);

    # Create df_tab_cum using helper function
    model_names = get_model_names()
    df_tab_cum = create_stacked_bar_weights(df_res_tmp, model_names, df_n_obs)

    labels_ = ["ZInf-NB", "ZInf-PLN", "ZInf-PLomax"]
    plot_stacked_bar(df_tab_cum, model_names;
        labels = labels_,
        legend_columns = 3,
        legend = (0, -0.15), #-0.22),
        bottom_margin = 15Plots.mm
    )
end