function read_master_with_fit_summary(; update_data = false)
	df_dds = CSV.read("../dt_surveys_master/master_dds.csv", DataFrame);
	df_obs = @pipe groupby(df_dds, [:key, :strat]) |>
				   combine(_, :y => sum => :n_answer);

	function summarise_data()
		df_sum = DataFrame()
		keys = df_dds[:, :key] |> unique
		for k in keys
			println(k)
			df_sum = vcat(df_sum, create_summary_stat_one_data(k))
		end
		CSV.write("../dt_intermediate/fit_summary.csv", df_sum)
	end

    if update_data == true
        # summarise_data()
    end

	df_sum = CSV.read("../dt_intermediate/fit_summary.csv", DataFrame)
    df_res = clean_df_res_dataframe(df_sum)
	return df_res
end

function filter_surveys(df_res::DataFrame)
    # Surveys with stratification by survey designs.
    lis_stratified = ["Danon 2013", "Leung 2017"]
    df_res_ana = @subset(df_res, @byrow (:key in lis_stratified) == false);
    return df_res_ana
end

function clean_df_res_dataframe(df_sum::DataFrame)
	df_res = flag_minimum_IC(df_sum, :waic);
	df_res[!, :model_abbr] = map(m -> model_abbr[m], df_res.model)
	df_res[!, :strat_pretty] = map(m -> setting_pretty[m], df_res.strat)
	df_res[!, :pc_error] = abs.(df_res.mean_raw .- df_res.mean) ./ df_res.mean .* 100
	df_res[!, :pc_error_pos_neg] = (df_res.mean_raw .- df_res.mean) ./ df_res.mean .* 100
    clean_survey_key_names!(df_res)
    return df_res
end

function plot_validation_est_vs_pc_error(df_pos::DataFrame;
	col_mean = :mean, col_error= :pc_error_pos_neg, strat = :strat_pretty,
    label = "mean", dx = 5.0)
	df_best = @subset(df_pos, :fmin_waic .== true)
	gdfs = groupby(df_best, strat)
	pls = []
	for (key, gdf) in zip(keys(gdfs), gdfs)
		m_max = maximum(gdf[:, col_mean]) + dx
		pl = plot(
            xlabel = "Estimated $label", ylabel = "Percent error (%)",
            xlims = (0, m_max), ylims = (-80, 80),
			yticks = (-100:20:100),
			)
		plot!(pl, [0, m_max], [0, 0], color = :black, label = "",
			title = key[strat], titlefontsize=12)
		@with gdf scatter!(pl, $col_mean, $col_error, group = :model_abbr, markerstrokewidth = 0.5)
		for r in eachrow(gdf)
			if isinf(r[col_error]) == true
				error("Check a row with Inf")
			end
            if (r.pc_error > 5.0)
                dx_anno = 0.0
                dy_anno = -5.0
				GR.setarrowsize(0.5)
				function draw_arrow(dx_anno, dy_anno, y_adj)
					plot!(pl,
						[r[col_mean] + dx_anno, r[col_mean]],
						[r[col_error] + dy_anno, r[col_error] + y_adj],
						arrow=arrow(:closed), color=:black, label="",
						linewidth=0.3, markersize=0.4)
				end

				if r.key == "CoMix UK"
					dx_anno = 0.0
					dy_anno = 20.0
					draw_arrow(dx_anno, dy_anno, 1.0)
				elseif r.key == "CoMix2 All"
					dx_anno = 2.0
					dy_anno = -20.0
					draw_arrow(dx_anno, dy_anno, -1.0)
				elseif r.key == "CoMix2 DK"
                dx_anno = 3.0
                dy_anno = -3.0

				elseif r.pc_error_pos_neg > 0
					dx_anno = 0.0
					dy_anno = 5.0

				end
				annotate!(pl, r[col_mean] + dx_anno, r[col_error] + dy_anno,
					text(r.key, :black, :center, 6))
			end
		end
        # pc_error lines
        plot!(pl, [0, m_max], [5, 5], color = :gray, linestyle = :dash, label = "±5%", alpha = 1.0)
        plot!(pl, [0, m_max], [-5, -5], color = :gray, linestyle = :dash, label = "", alpha = 1.0)
        plot!(pl, [0, m_max], [10, 10], color = :gray, linestyle = :dash, label = "±10%", alpha = 0.5)
        plot!(pl, [0, m_max], [-10, -10], color = :gray, linestyle = :dash, label = "", alpha = 0.5)
		push!(pls, pl)
	end
    plot!(pls[1], left_margin=10Plots.mm, top_margin=5Plots.mm)
    plot!(pls[2], right_margin=10Plots.mm)
    pos = (-0.1, 1.07)
    annotate!(pls[1], pos, text("A", :left, 18, "Helvetica"))
    annotate!(pls[2], pos, text("B", :left, 18, "Helvetica"))
	plot(pls..., layout = (1, 2), size = (800, 500), legend = :topright, format = :png)
end

function plot_validation_sample_vs_estimated(df_pos::DataFrame;
    col_raw = :mean_raw, col_est = :mean, strat = :strat_pretty,
    label = "mean", dx = 5.0)
    df_best = @subset(df_pos, :fmin_waic .== true)
    gdfs = groupby(df_best, strat)
    pls = []
    for (key, gdf) in zip(keys(gdfs), gdfs)
        m_max = maximum(gdf[:, col_raw]) + dx
        pl = plot(
            xlabel = "Sample $label", ylabel = "Estimated $label",
            xlims = (0, m_max), ylims = (0, m_max),
            )
        plot!(pl, [1, m_max], [1, m_max], color = :black, label = "",
            title = key[strat], titlefontsize=12)
        for r in eachrow(gdf)
            if isinf(r[col_est]) == true
                error("Check a row with Inf")
            end
            #if (r.model_abbr == "ZInf-PLomax")
            if (r.pc_error > 5.0)
                dx_anno = 1.0
                dy_anno = -1.0
                if (r.pc_error > 20.0) & (r.mean_raw < 10)
                    dx_anno = -1.2
                    dy_anno = 1.0
                elseif r.key == "Danon 2013 (paper)"
                    dx_anno = -5.0
                    dy_anno = 1.0
                elseif r.key == "Danon 2013 (online)"
                    dx_anno = -5.0
                    dy_anno = -1.0
                end
                annotate!(pl, r[col_raw] + dx_anno, r[col_est] + dy_anno,
                    text(r.key, :black, :left, 8))
            end
        end
        @with gdf scatter!(pl, $col_raw, $col_est, group = :model_abbr, markerstrokewidth = 0.5)
        # pc_error lines
        plot!(pl, [0, m_max], [0, m_max] .* 1.05, color = :gray, linestyle = :dash, label = "±5%", alpha = 1.0)
        plot!(pl, [0, m_max], [0, m_max] .* 0.95, color = :gray, linestyle = :dash, label = "", alpha = 1.0)
        plot!(pl, [0, m_max], [0, m_max] .* 1.10, color = :gray, linestyle = :dash, label = "±10%", alpha = 0.4)
        plot!(pl, [0, m_max], [0, m_max] .* 0.90, color = :gray, linestyle = :dash, label = "", alpha = 0.4)
        push!(pls, pl)
    end
    plot!(pls[1], left_margin=10Plots.mm, top_margin=5Plots.mm)
    plot!(pls[2], right_margin=10Plots.mm)
    pos = (-0.1, 1.07)
    annotate!(pls[1], pos, text("A", :left, 18, "Helvetica"))
    annotate!(pls[2], pos, text("B", :left, 18, "Helvetica"))
    plot(pls..., layout = (1, 2), size = (800, 500), legend = :topleft, format = :png)
end

###########################################
###### Validation of estimation methods ###
###########################################

function validation_PLN(d_true_PLN::PoissonLogNormal, n_sim)::DataFrame
    df_res = Vector{DataFrame}(undef, 2 * n_sim)
    Threads.@threads for ind in 1:n_sim
        data = rand(d_true_PLN, 100)
        # Hierarchical PoissonLogNormal
        model = model_hierarchical_PoissonLogNormal(data)
        chn = fit_model_with_forward_mode(model, 2000; progress=false)
        μ_obs, log_σ = extract_chain_info(chn)[1:2, "mean"]
        μ_ln, σ_ln = PoissonLogNormal_convert(μ_obs, log_σ)
        df_res[2 * ind - 1] = DataFrame(
            param1 = μ_ln, param2 = σ_ln,
            model="PLN_hierarchical", iter=ind
        )

        # Numerical PoissonLogNormal
        model = model_PoissonLogNormal(data |> DegreeDist)
        chn = fit_model_with_forward_mode(model, 2000; progress=false)
        μ_obs, log_σ = extract_chain_info(chn)[1:2, "mean"]
        μ_ln, σ_ln = PoissonLogNormal_convert(μ_obs, log_σ)
        df_res[2 * ind] = DataFrame(
            param1 = μ_ln, param2 = σ_ln,
            model="PLN_numerical", iter=ind
        )
    end
    return vcat(df_res...)
end

function validation_PLomax(d_true_PLomax::PoissonLomax, n_sim;
        n_sample=100)::DataFrame
    df_res = Vector{DataFrame}(undef, 2 * n_sim)
    Threads.@threads for ind in 1:n_sim
        data = rand(d_true_PLomax, n_sample)

        # Hierarchical PoissonLomax
        model = model_hierarchical_PoissonLomax(data)
        chn = fit_model_with_forward_mode(model, 2000; progress=false)
        log_α, log_β = extract_chain_info(chn)[1:2, "mean"]
        α = exp(log_α)
        β = exp(log_β)
        df_res[2 * ind - 1] = DataFrame(
            param1 = α, param2 = β,
            model="PLomax_hierarchical", iter=ind
        )

        # Numerical PoissonLomax
        model = model_PoissonLomax(data |> DegreeDist)
        chn = fit_model_with_forward_mode(model, 2000; progress=false)
        log_α, log_β = extract_chain_info(chn)[1:2, "mean"]
        α = exp(log_α)
        β = exp(log_β)
        df_res[2 * ind] = DataFrame(
            param1 = α, param2 = β,
            model="PLomax_numerical", iter=ind
        )
    end
    return vcat(df_res...)
end

function hv_plot!(pl::Plots.Plot, v; kwds...)
    plot!(pl, [0.0, v], [v, v]; kwds...)
    plot!(pl, [v, v], [0.0, v]; kwds...)
end

function plot_validation_PLN(df_res_PLN, μ_true, σ_true)
    df_p1 = unstack(df_res_PLN, :iter, :model, :param1)
    df_p2 = unstack(df_res_PLN, :iter, :model, :param2)
    p1_max = (df_p1[:, :PLN_hierarchical] |> maximum) + 1
    p2_max = (df_p2[:, :PLN_hierarchical] |> maximum) + 1
    pl1 = plot(xlabel="", ylabel="Estimates from Numerical",
        xlim=[0, p1_max], ylim=[0, p1_max])
    pl2 = plot(xlabel="",
        xlim=[0, p2_max], ylim=[0, p2_max])

    kwds_scatter = (markerstrokewidth= 0.5, legendfontsize=8)
    kwds_true = (ls=:dash, color=:green, label="")
    scatter!(pl1, df_p1[:, :PLN_hierarchical], df_p1[:, :PLN_numerical];
        label="PLN: μ", kwds_scatter...)
    plot!(pl1, [0.0, p1_max], [0.0, p1_max], ls=:solid, color=:black, label="")
    hv_plot!(pl1, μ_true; kwds_true...)
    scatter!(pl2, df_p2[:, :PLN_hierarchical], df_p2[:, :PLN_numerical];
        label="PLN: σ", kwds_scatter...)
    plot!(pl2, [0.0, p2_max], [0.0, p2_max], ls=:solid, color=:black, label="")
    hv_plot!(pl2, σ_true; kwds_true...)
    return (pl1, pl2)
end

function plot_validation_PLomax(df_res_PLomax, α_true, β_true)
    df_p1 = unstack(df_res_PLomax, :iter, :model, :param1)
    df_p2 = unstack(df_res_PLomax, :iter, :model, :param2)
    p1_max = (df_p1[:, :PLomax_hierarchical] |> maximum) + 1
    p2_max = (df_p2[:, :PLomax_hierarchical] |> maximum) + 1
    pl1 = plot(xlabel="Estimates from hierarchical", ylabel="Estimates from Numerical",
        xlim=[0, p1_max], ylim=[0, p1_max])
    pl2 = plot(xlabel="Estimates from hierarchical",
        xlim=[0, p2_max], ylim=[0, p2_max])

    kwds_scatter = (markerstrokewidth= 0.5, legendfontsize=8)
    kwds_true = (ls=:dash, color=:green, label="")
    scatter!(pl1, df_p1[:, :PLomax_hierarchical], df_p1[:, :PLomax_numerical];
        label="PLomax: α", kwds_scatter...)
    plot!(pl1, [0.0, p1_max], [0.0, p1_max], ls=:solid, color=:black, label="")
    hv_plot!(pl1, α_true; kwds_true...)
    scatter!(pl2, df_p2[:, :PLomax_hierarchical], df_p2[:, :PLomax_numerical];
        label="PLomax: β", kwds_scatter...)
    plot!(pl2, [0.0, p2_max], [0.0, p2_max], ls=:solid, color=:black, label="")
    hv_plot!(pl2, β_true; kwds_true...)
    return (pl1, pl2)
end
