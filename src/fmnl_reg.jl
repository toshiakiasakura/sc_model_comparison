"""Fractional multinomial distributions.
"""
@model function model_fmnl(x::Matrix, y::Matrix)
	n_x, n_col_x = size(x)
	n_col_y = size(y)[2]
	c ~ filldist(Normal(0, 3), n_col_y - 1) # Fixed term.
	β ~ filldist(Normal(0, 3), n_col_x, n_col_y - 1) # For explanatory variables.
	for i in 1:n_x
		# The last outcome is scaled to one.
		scale = 1
		for j in 1:(n_col_y-1)
			log_scale = c[j] + sum(x[i, :] .* β[:, j])
			scale += exp(log_scale)
		end
		# the last outcome is scaled to one, and cancelled out.
		ll = 0.0
		for j in 1:(n_col_y-1)
			ll += y[i, j] * (c[j] + sum(x[i, :] .* β[:, j]))
		end
		Turing.@addlogprob! ll - log(scale)
	end
end

function calculate_fmnl_probs(x::AbstractMatrix, c::AbstractVector, β::AbstractMatrix)
    n_x = size(x)[1]
    n_col_y = length(c) + 1 # Total categories
    probs = zeros(n_x, n_col_y)

    for i in 1:n_x
        log_scales = zeros(n_col_y - 1)
        for j in 1:(n_col_y - 1)
            log_scales[j] = c[j] + sum(x[i, :] .* β[:, j])
        end
        scale = 1.0 + sum(exp, log_scales)

        for j in 1:(n_col_y - 1)
            probs[i, j] = exp(log_scales[j]) / scale
        end
        probs[i, n_col_y] = 1.0 / scale
    end
    return probs
end