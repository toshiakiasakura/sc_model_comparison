###############################
##### Degree distribution #####
###############################

function plot_ccdf_across_surveys(
    df_dds::DataFrame, keys, strat::String, txt::String;
    ano_pos = (-0.12, 1.03), fontsize=16, kwds...)
    df_tmp = @subset(df_dds, :strat .== strat)

    xtk = ([1, 10, 100, 1000, 10_000], [L"1", L"10", L"10^{2}", L"10^{3}", L"10^{4}"])
    pl = plot(; xaxis = :log10, ylim = [-5, 0], xlim=[1, 10_000], xticks = xtk,
        kwds...)
    for k in keys
        dd = @subset(df_tmp, :key .== k) |> DegreeDist
        plot_ccdf!(pl, dd; label=k, markersize=2.5, markerstrokewidth = 0.0)
    end
    annotate!(pl, ano_pos, text(txt, :black, fontsize, "Helvetica"))
    pl
end

function plot_all_deg_and_separate_subplots(df_dds)
    keys = ["CoMix UK", "CoMix2 All", "Danon 2013 (paper)", "Willem 2012", "Grijalva 2015"]

    pl = plot_ccdf_across_surveys(df_dds, keys, "all", "A")
    pl1 = plot_all_hm_nhm(df_dds, "CoMix UK"; panel_name="B")
    pl2 = plot_all_hm_nhm(df_dds, "CoMix2 All"; panel_name="C")
    pl3 = plot_all_hm_nhm(df_dds, "Grijalva 2015"; panel_name="D")
    pl4 = plot_all_hm_nhm(df_dds, "Willem 2012"; panel_name="E")
    pl5 = plot_all_hm_nhm(df_dds, "Danon 2013 (paper)"; panel_name="F")

    xlabel = "Number of contacts per day"
    plot!(pl, ylabel = "CCDF")
    plot!(pl3, ylabel = "CCDF", xlabel = xlabel)
    plot!(pl4, xlabel = xlabel)
    plot!(pl5, xlabel = xlabel)

    pls = [pl, pl1, pl2, pl3, pl4, pl5]
    h = []
    layout = @layout [a{0.66w, 0.66h} [b; c]; [d e] f]
    plot(pls..., layout = layout,
        size = (800, 800),
		right_margin = 5Plots.mm, top_margin=5Plots.mm,
        dpi=300)
end

function plot_all_hm_nhm(df_dds::DataFrame, key::String; panel_name = "", color = 1, kwds...)
	df_tmp = @subset(df_dds, :key .== key)
	dd_all = @subset(df_tmp, :strat .== "all") |> DegreeDist
	dd_hm = @subset(df_tmp, :strat .== "home") |> DegreeDist
	dd_nhm = @subset(df_tmp, :strat .== "non-home") |> DegreeDist;

	pl = plot(;
		xaxis = :log10, size = (500, 500),
		xlim = [1, 100], ylim = [-1.5, 0.1],
		legend = (0.1, 0.2),
		xticks = ([1, 10, 100], [L"1", L"10", L"10^{2}"]),
	)
	kwds1 = (markersize = 2.5, markerstrokewidth = 0, lw = 1.5, ytk_digit=2)
	kwds2 = (markersize = 2.5, markerstrokewidth = 0, ytk_digit=2)
	plot_ccdf!(pl, dd_all; color = 14, ls = :solid, label = "All", kwds1...)
	plot_ccdf!(pl, dd_hm; color = 9, ls = :solid, label = "Home", kwds2...)
	plot_ccdf!(pl, dd_nhm; color = 16, ls = :solid, label = "Non-Home", kwds2...)
	annotate!(pl, (0.8, 0.9), text(key, :black, 10, "Helvetica"))
	annotate!(pl, (-0.10, 1.08), text(panel_name, :black, 16, "Helvetica"))
	plot!(pl; kwds...)
	pl
end

function plot_ccdf_across_surveys_with_different_setting(df_dds::DataFrame)
    # Key definition
    rem_keys = ["Danon 2013", "Leung 2017"]
    comix2_keys = [
        "CoMix2 All", "CoMix2 GR", "CoMix2 IT", "CoMix2 PT",
        "CoMix2 HR",  "CoMix2 EE", "CoMix2 AT", "CoMix2 DK",
        "CoMix2 BE", "CoMix2 PL",
    ]
    large_keys = [
        "CoMix UK", "Mossong 2008",
        "Dodd 2016", "Beraud 2015", "Read 2014", "Willem 2012",
    ]
    small_keys = [
        "Melegaro 2017", "Hens 2009", "Zhang 2019", "Horby 2011",
        "Kassteele 2017", "Wirya 2020", "Grijalva 2015",
    ]
    po_keys = [
        "Danon 2013 (paper)", "Danon 2013 (online)",
        "Leung 2017 (online)", "Leung 2017 (paper)",
    ]
    all_keys = vcat(comix2_keys, rem_keys, large_keys, small_keys, po_keys)
    @test length(all_keys) == 29

    kwds = (; size=(600, 500), dpi=300,
        left_margin=5Plots.mm, bottom_margin=5Plots.mm,
        legendfontsize=10, tickfontsize=12, labelfontsize=12)
    key_pairs = [comix2_keys, large_keys, small_keys, po_keys]
    panels = "ABCDEFGHIJKL"
    fontsize = 24
    ind = 1
    for strat in ["all", "home", "non-home"]
        for keys_ in key_pairs
            pl = plot_ccdf_across_surveys(df_dds, keys_, strat, string(panels[ind]);
                ano_pos=(0.1, 0.1), fontsize=fontsize, kwds...)

            xfont = mod(ind, 4) == 0 ? font(:black) : font(:white)
            yfont = ind <= 4 ? font(:black) : font(:white)
            plot!(pl, xlabel="Number of contacts per day", ylabel="CCDF",
                xguidefont=xfont, yguidefont=yfont)
            if mod(ind, 4) == 1
                annotate!(pl, (0.5, 1.05),
                    text(uppercasefirst(strat), :black, fontsize, "Helvetica"))
                plot!(pl, top_margin=10Plots.mm)
            end
            savefig(pl, "../fig/degree_dist$(ind).png")
            ind += 1
        end
    end
    imgs = []
    for i in 1:12
        path = "../fig/degree_dist$(i).png"
        img = @pipe load(path) |> imresize(_, ratio=0.33)
        push!(imgs, img)
        rm(path)
    end
    v1, v2, v3 = vcat(imgs[1:4]...), vcat(imgs[5:8]...), vcat(imgs[9:12]...)
    combined = hcat(v1, v2, v3)
    save("../fig/degree_dist_all_combined.png", combined)
    combined
end
