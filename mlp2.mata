*! version 1.0.0  22nov2017
////////////////////////////////////////////////////////////////////////////////

local RS real scalar
local SS string scalar
local SRV string vector
local RV real vector
local RM real matrix
local CV real colvector
local MPCV pointer(real colvector) matrix

mata:

////////////////////////////////////////////////////////////////////////////////
// class c_mlp2

class c_mlp2 {

protected:
	`RS' n, p, m1, m2, mout
	// data
	`RM' y, X
	`SS' xvars
	`SRV' ylabels

	// rescaling parameters
	`RV' means
	`RV' vars
	
	// coefficients
	`RM' alpha, beta, gamma
	// biases
	`RM' alpha0, beta0, gamma0
	// complement dropout indices
	`RV' indLayer0, indLayer1, indLayer2
	`RV' dropLayer1, dropLayer2
	
	// placeholders
	`RM' U, V, UU, VV, Z, expZ, rexpZ
	`RM' alphaGrad,  betaGrad,  gammaGrad
	`RM' alpha0Grad, beta0Grad, gamma0Grad
	`RM' dGdV, dGdU
	`MPCV' U0, V0
	
	// training operations	
	`RS' train_time, train_epochs
	`RS' train_loss
	`RS' nobias
	`RS' dropout_p
	`RS' echo
	
	void _init()
	virtual void _init_extra()

	void _compute_forward()
	`RS' _compute_loss()

	void _grad_alpha()
	void _grad_beta()
	void _grad_gamma()

	void _dropout()
	
	void _gradient_test()

public:
	void new()
	void save_params()
	void restore_params()
	void predict()
	void simulate()
	void score()
}

void c_mlp2::new()
{
	y = J(1,1,.)
	X = J(1,1,.)
	n = 0
	p = 0
	m1 = 0 
	m2 = 0
	mout = 0
	alpha  = J(1,1,.)
	beta   = J(1,1,.)
	gamma  = J(1,1,.)
	alpha0 = J(1,1,0)
	beta0  = J(1,1,0)
	gamma0 = J(1,1,0)
	indLayer0 = J(1, 0, 0)
	indLayer1 = J(1, 0, 0)
	indLayer2 = J(1, 0, 0)
	dropLayer1 = J(1, 0, 0)
	dropLayer2 = J(1, 0, 0)
	train_time   = 0
	train_epochs = 0
	nobias = 0
	dropout_p = 0
	echo = 0
}

void c_mlp2::save_params()
{
	st_eclear()
	st_matrix("e(means)",  means,  "hidden")
	st_matrix("e(vars)",   vars,   "hidden")
	st_matrix("e(alpha)",  alpha,  "visible")
	st_matrix("e(beta)",   beta,   "visible")
	st_matrix("e(gamma)",  gamma,  "visible")
	if (!nobias) {
		st_matrix("e(alpha0)", alpha0, "visible")
		st_matrix("e(beta0)",  beta0,  "visible")
		st_matrix("e(gamma0)", gamma0, "visible")
	}
	st_numscalar("e(train_time)",   train_time,   "hidden")
	st_numscalar("e(train_epochs)", train_epochs, "hidden")
	st_numscalar("e(train_loss)",   train_loss,   "hidden")
}

void c_mlp2::restore_params()
{
	means  = st_matrix("e(means)")
	vars   = st_matrix("e(vars)")
	alpha  = st_matrix("e(alpha)")
	beta   = st_matrix("e(beta)")
	gamma  = st_matrix("e(gamma)")
	p = rows(alpha)
	m1   = cols(alpha)
	m2   = cols(beta)
	mout = cols(gamma)
	if (p < 1 || m1 < 1 || m2 < 1 || mout < 1) {
		errprintf("Invalid mlp2 parameters")
		exit(111)
	}
	if (rows(beta) != m1 || rows(gamma) != m2) {
		errprintf("Incompatible mlp2 parameters")
		exit(503)
	}
	if (length(means) != p) {
		errprintf("Incompatible mean parameters")
		exit(503)
	}
	if (length(vars) != p) {
		errprintf("Incompatible var parameters")
		exit(503)
	}
	alpha0 = st_matrix("e(alpha0)")
	if (length(alpha0) == 0) {
		alpha0 = J(1, m1, 0)
	}
	beta0  = st_matrix("e(beta0)")
	if (length(beta0) == 0) {
		beta0 = J(1, m2, 0)
	}
	gamma0 = st_matrix("e(gamma0)")
	if (length(gamma0) == 0) {
		gamma0 = J(1, mout, 0)
	}
	if (length(alpha0) != m1) {
		errprintf("Incompatible alpha0 parameters")
		exit(503)
	}
	if (length(beta0) != m2) {
		errprintf("Incompatible beta0 parameters")
		exit(503)
	}
	if (length(gamma0) != mout) {
		errprintf("Incompatible gamma0 parameters")
		exit(503)
	}
}

void c_mlp2::_dropout()
{
	`RS' j
	`CV' ind

	if (dropout_p <= 0 || p < 1 || m1 < 1 || m2 < 1) {
		return
	}

	// dropout applies to the lowest 3 layers
	ind = runiform(1, p)
	indLayer0  = selectindex(ind :>= dropout_p)
	ind = runiform(1, m1)
	dropLayer1 = selectindex(ind :< dropout_p)
	indLayer1  = selectindex(ind :>= dropout_p)
	ind = runiform(1, m2)
	dropLayer2 = selectindex(ind :< dropout_p)
	indLayer2  = selectindex(ind :>= dropout_p)
}

void c_mlp2::predict(`SS' sxvars, `SS' stouse, `SS' sgenvar)
{
	`RS' k
	`RM' touse
	
	touse = 0
	st_view(touse, ., stouse)
	touse = selectindex(touse)
	
	restore_params()
	
	stata("local __ylabels = e(ylabels)", 0)
	ylabels = tokens(st_local("__ylabels"))
	if (length(ylabels) != mout) {
		ylabels = J(1, mout, "")
		for (k = 1; k <= mout; k++) {
			ylabels[k] = sprintf("%s_%g", sgenvar, k)
		}
	}
	else {
		for (k = 1; k <= mout; k++) {
			ylabels[k] = sprintf("%s_%s", sgenvar, ylabels[k])
		}
	}

	X = 0
	st_view(X, touse, sxvars)
	n = rows(X)
	if (cols(X) != p) {
		errprintf("Incompatible variable list {bf:%s}\n", sxvars)
		exit(503)
	}
	
	// rescale test dataset using training means and vars
	X = ((X :- means) :/ sqrt(vars))
	
	_compute_forward(J(1,0,0))

	for (k = 1; k <= mout; k++) {
		stata(sprintf("cap drop %s", ylabels[k]))
		stata(sprintf("generate float %s = .", ylabels[k]), 1)
		st_store(touse, ylabels[k], expZ[,k])
	}
}

void c_mlp2::simulate(`SS' sxvars, `SS' stouse, `SS' sgenvar)
{
	`RS' i
	`RM' touse
	`CV' ygen
	
	touse = 0
	st_view(touse, ., stouse)
	touse = selectindex(touse)
	
	restore_params()

	X = 0
	st_view(X, touse, sxvars)
	n = rows(X)
	if (cols(X) != p) {
		errprintf("Incompatible variable list {bf:%s}\n", sxvars)
		exit(503)
	}

	// rescale test dataset using training means and vars
	X = ((X :- means) :/ sqrt(vars))
	
	_compute_forward(J(1,0,0))

	stata(sprintf("cap drop %s", sgenvar), 1)
	stata(sprintf("generate double %s = .", sgenvar), 1)
	ygen = J(rows(expZ), 1, .)
	for (i = 1; i <= rows(expZ); i++) {
		ygen[i,] = rdiscrete(1, 1, expZ[i,])
	}
	st_store(touse, sgenvar, ygen)
}

void c_mlp2::score(`SS' sy, `SS' stouse)
{
	`RS' k
	`RM' touse
	
	touse = 0
	st_view(touse, ., stouse)
	touse = selectindex(touse)

	y = st_data(touse, sy)
	if (rows(y) != n) {
		errprintf("Incompatible label variable {bf:%s}\n", sy)
		exit(503)
	}
	for (k = 1; k <= length(y); k++) {
		y[k] = expZ[k,y[k]]
	}
	st_numscalar("e(score_meanprob)", mean(y), "hidden")
	
	y = st_data(touse, sy)
	for (k = 1; k <= length(y); k++) {
		y[k] = expZ[k,y[k]] >= max(expZ[k,])
	}
	st_matrix("e(pred_err_ind)", selectindex(y:==0)', "hidden")
	st_numscalar("e(score_acc)", mean(y), "hidden")
}

void c_mlp2::_init(`SS' sy, `SS' sxvars, `SS' stouse, `RS' rm1, `RS' rm2, `RS' rmout)
{
	`RM' touse
	
	m1   = rm1
	m2   = rm2
	mout = rmout
	xvars = sxvars

	touse = 0
	st_view(touse, ., stouse)
	touse = selectindex(touse)

	y = 0
	st_view(y, touse, sy)

	X = 0
	st_view(X, touse, sxvars)
	n = rows(X)
	p = cols(X)
	if (rows(y) != n) {
		exit(503)
	}

	alpha = rnormal(p,  m1,   0, sqrt(2/(p+m1)))
	beta  = rnormal(m1, m2,   0, sqrt(2/(m1+m2)))
	gamma = rnormal(m2, mout, 0, sqrt(2/(m2+mout)))
	if (!nobias) {
		alpha0 = rnormal(1, m1,   0, sqrt(2/(m1)))
		beta0  = rnormal(1, m2,   0, sqrt(2/(m2)))
		gamma0 = rnormal(1, mout, 0, sqrt(2/(mout)))
		alpha0 = J(1, m1,   0)
		beta0  = J(1, m2,   0)
		gamma0 = J(1, mout, 0)
	}
	else {
		alpha0 = J(1, m1,   0)
		beta0  = J(1, m2,   0)
		gamma0 = J(1, mout, 0)
	}

	U  = J(1, m1, 0)
	UU = J(1, m1, 0)
	U0 = J(1, m1, NULL)
	V  = J(1, m2, 0)
	VV = J(1, m2, 1)
	V0 = J(1, m2, NULL)
	Z  = J(1, mout, 0)
	expZ  = exp(Z)
	rexpZ = rowsum(expZ)

	alphaGrad  = J(0,0,0)
	betaGrad   = J(0,0,0)
	gammaGrad  = J(0,0,0)
	alpha0Grad = J(0,0,0)
	beta0Grad  = J(0,0,0)
	gamma0Grad = J(0,0,0)
	dGdV = J(0,0,0)
	dGdU = J(0,0,0)
}

void c_mlp2::_init_extra()
{
}

void c_mlp2::_compute_forward(`RV' batch_ind)
{
	`RS' j
	`CV' sind

	if (length(batch_ind) >= 1) {
		if (dropout_p > 0 && cols(indLayer0) > 0 && 
			cols(indLayer0) < p) {
			U  = X[batch_ind, indLayer0] * alpha[indLayer0',] :+ alpha0
		}
		else {
			U  = X[batch_ind,] * alpha :+ alpha0
		}
	}
	else {
		if (dropout_p > 0 && cols(indLayer0) > 0 && 
			cols(indLayer0) < p) {
			U  = X[, indLayer0] * alpha[indLayer0',] :+ alpha0
		}
		else {
			U  = X * alpha :+ alpha0
		}
	}

	UU = U
	U0 = J(1, cols(U), NULL)
	for (j = 1; j <= cols(U); j++) {
		sind   = selectindex(U[,j] :<= 0)
		U0[j]  = &J(1,0,0)
		*U0[j] = sind
		if (length(sind) > 0) {
			UU[sind,j] = J(length(sind), 1, 0)
		}
	}

	if (dropout_p > 0 && cols(indLayer1) > 0 && cols(indLayer1) < m1) {
		V = UU[, indLayer1] * beta[indLayer1',] :+ beta0
	}
	else {
		V = UU * beta :+ beta0
	}

	VV = V
	V0 = J(1, cols(V), NULL)
	for (j = 1; j <= cols(V); j++) {
		sind   = selectindex(V[,j] :<= 0)
		V0[j]  = &J(1,0,0)
		*V0[j] = sind
		if (length(sind) > 0) {
			VV[sind,j] = J(length(sind), 1, 0)
		}
	}

	if (dropout_p > 0 && cols(indLayer2) > 0 && cols(indLayer2) < m2) {
		Z = VV[, indLayer2] * gamma[indLayer2',] :+ gamma0
	}
	else {
		Z = VV * gamma :+ gamma0
	}

	// if there is z > 709, exp(z) returns .
	expZ  = exp(Z)
	rexpZ = rowsum(expZ)
	expZ  = expZ  :/ rexpZ
}

`RS' c_mlp2::_compute_loss(`RV' batch_ind)
{
	`RS' s, loss
	loss = 0
	if (length(batch_ind) >= 1) {
		for (s = 1; s <= length(batch_ind); s++) {
			loss = loss + Z[s, y[batch_ind[s]]]
		}
	}
	else {
		for (s = 1; s <= rows(Z); s++) {
			loss = loss + Z[s, y[s]]
		}
	}

	return ((sum(log(rexpZ)) - loss)/rows(rexpZ))
}

void c_mlp2::_grad_gamma(`RV' batch_ind) 
{
	`RS' k, s, ys, nfact
	`CV' sind
	// gammaGrad: m2 x mout
	gammaGrad  = VV' * expZ
	if (!nobias) {
		gamma0Grad = colsum(expZ)
	}
	// dGdV: n x m2
	dGdV = expZ * gamma'
	if (length(batch_ind) >= 1) {
		for (s = 1; s <= length(batch_ind); s++) {
			ys = y[batch_ind[s]]
			gammaGrad[.,ys]  = gammaGrad[.,ys] - VV[s,.]'
			if (!nobias) {
				gamma0Grad[.,ys] = gamma0Grad[.,ys] - 1
			}
			dGdV[s,.] = dGdV[s,.] - gamma[.,ys]'
		}
	}
	else {
		for (s = 1; s <= rows(y); s++) {
			ys = y[s]
			gammaGrad[.,ys]  = gammaGrad[.,ys] - VV[s,.]'
			if (!nobias) {
				gamma0Grad[.,ys] = gamma0Grad[.,ys] - 1
			}
			dGdV[s,.] = dGdV[s,.] - gamma[.,ys]'
		}
	}
	for (k = 1; k <= length(V0); k++) {
		sind = *(V0[k])
		dGdV[sind, k] = J(length(sind), 1, 0)
	}
	nfact = 1 / rows(expZ)
	gammaGrad  = gammaGrad  :* nfact
	dGdV = dGdV :* nfact
	if (!nobias) {
		gamma0Grad = gamma0Grad :* nfact
	}
}

void c_mlp2::_grad_beta() 
{
	`RS' j, k
	`CV' sind
	// betaGrad: m1 x m2
	betaGrad  =  UU' * dGdV
	// dGdU: n x m1
	dGdU = dGdV * beta'
	for (j = 1; j <= length(U0); j++) {
		sind = *(U0[j])
		dGdU[sind, j] = J(length(sind), 1, 0)
	}
	if (dropout_p > 0 && cols(dropLayer2) > 0) {
		betaGrad[, dropLayer2] = J(rows(betaGrad), cols(dropLayer2), 0)
	}

	if (nobias) {
		return
	}
	// beta0Grad: 1 x m2
	beta0Grad =  dGdV
	for (k = 1; k <= length(V0); k++) {
		sind = *(V0[k])
		beta0Grad[sind, k] = J(length(sind), 1, 0)
	}
	beta0Grad = colsum(beta0Grad)
}

void c_mlp2::_grad_alpha(`RV' batch_ind) 
{
	`RS' j
	`CV' sind
	// alphaGrad: p x m1
	if (length(batch_ind) >= 1) {
		alphaGrad =  X[batch_ind,]' * dGdU
	}
	else {
		alphaGrad =  X' * dGdU
	}
	if (dropout_p > 0 && cols(dropLayer1) > 0) {
		alphaGrad[, dropLayer1] = J(rows(alphaGrad), cols(dropLayer1), 0)
	}
	if (nobias) {
		return
	}
	// alpha0Grad: 1 x m1
	alpha0Grad =  dGdU
	for (j = 1; j <= length(U0); j++) {
		sind = *(U0[j])
		alpha0Grad[sind, j] = J(length(sind), 1, 0)
	}
	alpha0Grad = colsum(alpha0Grad)
}


void c_mlp2::_gradient_test(`RS' delta, 
	`RS' ialpha, `RS' jalpha,
	`RS' ibeta,  `RS' jbeta,
	`RS' igamma, `RS' jgamma)
{
	`RM' alpha2, beta2, gamma2
	`RM' alphaGrad2, betaGrad2, gammaGrad2
	`RS' curloss, steploss
	`RV' batch_ind

	alpha2 = alpha
	beta2  = beta
	gamma2 = gamma

	batch_ind = 1..n

	_compute_forward(batch_ind)
	curloss = _compute_loss(batch_ind)

	_grad_gamma(batch_ind)
	_grad_beta()
	_grad_alpha(batch_ind)

	alphaGrad2 = alphaGrad
	betaGrad2  = betaGrad
	gammaGrad2 = gammaGrad
	
	if (delta <= 0) {
		delta = 1e-8
	}

	alpha[ialpha, jalpha] = alpha[ialpha, jalpha] + delta
	_compute_forward(batch_ind)
	steploss = _compute_loss(batch_ind)
	str = sprintf("alpha[%g, %g]: %g, %g", ialpha, jalpha, 
		alphaGrad2[ialpha, jalpha], (steploss-curloss)/delta)
	str
	alpha = alpha2

	beta[ibeta, jbeta] = beta[ibeta, jbeta] + delta
	_compute_forward(batch_ind)
	steploss = _compute_loss(batch_ind)
	str = sprintf("beta[%g, %g]: %g, %g", ibeta, jbeta, 
		betaGrad2[ibeta, jbeta], (steploss-curloss)/delta)
	str
	beta = beta2

	gamma[igamma, jgamma] = gamma[igamma, jgamma] + delta
	_compute_forward(batch_ind)
	steploss = _compute_loss(batch_ind)
	str = sprintf("gamma[%g, %g]: %g, %g", igamma, jgamma, 
		gammaGrad2[igamma, jgamma], (steploss-curloss)/delta)
	str
	gamma = gamma2
}

////////////////////////////////////////////////////////////////////////////////
// class c_mlp2_optimizer

class c_mlp2_optimizer extends c_mlp2 {
protected:
	`RS' iter

	virtual void _update()
public:
	void fit()
}

void c_mlp2_optimizer::fit(`SS' sy, `SS' sxvars, `SS' touse, 
	`RS' rm1, `RS' rm2, `RS' rmout, 
	`RS' batchsize, `RS' epochs, `RS' losstol, `RS' dropout, 
	`RS' snobias, `RS' secho)
{
	`RS' curloss, lastloss, batchloss, brkval
	`RS' nbatches, nb, ntotal
	`RV' batch_ind, sample_ind, ind

	echo      = secho
	nobias    = snobias
	dropout_p = dropout

	// start the timer
	timer_clear(100)
	timer_on(100)

	_init(sy, sxvars, touse, rm1, rm2, rmout)

	_init_extra()

	means = mean(X)
	vars  = diagonal(variance(X))'
	ind   = selectindex(vars :<= 0)
	vars[ind] = J(1, length(ind), 1)
	ind   = selectindex(vars :>= .)
	if (length(ind) > 0) {
		means[ind] = J(1, length(ind), 0)
		vars[ind]  = J(1, length(ind), 1)
	}

	// rescale training dataset
	X = ((X :- means) :/ sqrt(vars))

	if (batchsize <= 0 || batchsize > n) {
		batchsize = n;
	}
	nbatches = floor(n/batchsize)

	brkval = setbreakintr(1)

	lastloss = .
	for (iter = 1; iter <= epochs; iter++) {

		sample_ind = order(runiform(n, 1), 1)

		curloss = 0
		ntotal  = 0
		for (nb = 1; nb <= nbatches; nb++) {
		
			if (breakkey()) {
				break
			}

			if (batchsize < n) {
				if (nb < nbatches) {
					batch_ind = (1+(nb-1)*batchsize)..(nb*batchsize)
				}
				else {
					batch_ind = (1+(nb-1)*batchsize)..(n)
				}
			}
			else {
				batch_ind = 1..n
			}

			batch_ind = sample_ind[batch_ind]

			_dropout()

			_compute_forward(batch_ind)

			batchloss = _compute_loss(batch_ind)

			curloss = curloss + length(batch_ind) * batchloss
			ntotal = ntotal + length(batch_ind)
			if (curloss >= .) {
				break
			}

			_update(batch_ind)
		}
		curloss = curloss / ntotal

		if (breakkey()) {
			breakkeyreset()
			break
		}

		if (echo > 0 && echo*floor((iter-1)/echo)+1 == iter) {
			printf("%g: %g\n", iter, curloss)
		}
		if (curloss >= .) {
			errprintf("Fail to converge.\n")
			printf("You may need to reduce the learning rate or choose different optimizer.\n")
			break
		}
		if (abs(lastloss - curloss) < losstol) {
			break
		}
		lastloss  = curloss
	}

	//_gradient_test(1e-6, 3, 1, 1, 2, 2, 1)

	brkval = setbreakintr(brkval)

	train_loss   = curloss
	train_epochs = iter
	
	timer_off(100)
	train_time = timer_value(100)[1,1]
	timer_clear(100)
}

////////////////////////////////////////////////////////////////////////////////
// class c_mlp2_gd

class c_mlp2_gd extends c_mlp2_optimizer {
protected:
	`RS' lrate, gainrate
	`RS' friction, friction0

	`RM' alphaGain,  betaGain,  gammaGain
	`RM' alpha0Gain, beta0Gain, gamma0Gain
	
	`RM' alphaGrad1,  betaGrad1,  gammaGrad1
	`RM' alpha0Grad1, beta0Grad1, gamma0Grad1

	void _update()
	
	void _init_gain()
	void _update_gain()
	
public:
	void new()
	void set_lrate()
	void set_gainrate()
	void set_friction()
}


void c_mlp2_gd::new()
{
	lrate     = 0.01
	gainrate  = 0.05
	friction0 = 0
	
	alphaGain = J(0, 0, 1)
	betaGain  = J(0, 0, 1)
	gammaGain = J(0, 0, 1)
	alpha0Gain = J(1, 0, 1)
	beta0Gain  = J(1, 0, 1)
	gamma0Gain = J(1, 0, 1)
}

void c_mlp2_gd::set_lrate(`RS' rate)
{
	lrate = rate
}

void c_mlp2_gd::set_gainrate(`RS' rate)
{
	gainrate = rate
	if (gainrate < 0) gainrate = 0
}

void c_mlp2_gd::set_friction(`RS' fric)
{
	friction0 = fric
}

void c_mlp2_gd::_init_gain()
{
	alphaGain = J(p,  m1,   1)
	betaGain  = J(m1, m2,   1)
	gammaGain = J(m2, mout, 1)
	if (!nobias) {
		alpha0Gain = J(1, m1,   1)
		beta0Gain  = J(1, m2,   1)
		gamma0Gain = J(1, mout, 1)
	}
}

void c_mlp2_gd::_update_gain()
{
	`RS' j
	`SM' sind

	if (gainrate <= 0) 
		return
	
	if (	length(alphaGain) == 0 ||
		length(betaGain)  == 0 ||
		length(gammaGain) == 0) {
		_init_gain()
	}

	if (	length(alphaGrad1) != length(alphaGrad) ||
		length(betaGrad1)  != length(betaGrad)  ||
		length(gammaGrad1) != length(gammaGrad)) {
		alphaGrad1 = alphaGrad
		betaGrad1  = betaGrad
		gammaGrad1 = gammaGrad
		if (!nobias) {
			alpha0Grad1 = alpha0Grad
			beta0Grad1  = beta0Grad
			gamma0Grad1 = gamma0Grad
		}
		return
	}

	alphaGrad1 = alphaGrad1 :* alphaGrad
	for (j = 1; j <= cols(alphaGrad); j++) {
		sind = selectindex(alphaGrad1[,j] :> 0)
		if (length(sind) > 0) {
			alphaGain[sind,j] = alphaGain[sind,j] :+ gainrate
		}
		sind = selectindex(alphaGrad1[,j] :< 0)
		if (length(sind) > 0) {
			alphaGain[sind,j] = alphaGain[sind,j] :* (1-gainrate)
		}
	}
	alphaGrad1 = alphaGrad

	betaGrad1 = betaGrad1 :* betaGrad
	for (j = 1; j <= cols(betaGrad); j++) {
		sind = selectindex(betaGrad1[,j] :> 0)
		if (length(sind) > 0) {
			betaGain[sind,j] = betaGain[sind,j] :+ gainrate
		}
		sind = selectindex(betaGrad1[,j] :< 0)
		if (length(sind) > 0) {
			betaGain[sind,j] = betaGain[sind,j] :* (1-gainrate)
		}
	}
	betaGrad1 = betaGrad

	gammaGrad1 = gammaGrad1 :* gammaGrad
	for (j = 1; j <= cols(gammaGrad); j++) {
		sind = selectindex(gammaGrad1[,j] :> 0)
		if (length(sind) > 0) {
			gammaGain[sind,j] = gammaGain[sind,j] :+ gainrate
		}
		sind = selectindex(gammaGrad1[,j] :< 0)
		if (length(sind) > 0) {
			gammaGain[sind,j] = gammaGain[sind,j] :* (1-gainrate)
		}
	}
	gammaGrad1 = gammaGrad

	if (!nobias) {
		alpha0Grad1 = alpha0Grad1 :* alpha0Grad
		sind = selectindex(alpha0Grad1[1,] :> 0)
		if (length(sind) > 0) {
			alpha0Gain[1,sind] = alpha0Gain[1,sind] :+ gainrate
		}
		sind = selectindex(alpha0Grad1[1,] :< 0)
		if (length(sind) > 0) {
			alpha0Gain[1,sind] = alpha0Gain[1,sind] :* (1-gainrate)
		}
		alpha0Grad1 = alpha0Grad
		
		beta0Grad1 = beta0Grad1 :* beta0Grad
		sind = selectindex(beta0Grad1[1,] :> 0)
		if (length(sind) > 0) {
			beta0Gain[1,sind] = beta0Gain[1,sind] :+ gainrate
		}
		sind = selectindex(beta0Grad1[1,] :< 0)
		if (length(sind) > 0) {
			beta0Gain[1,sind] = beta0Gain[1,sind] :* (1-gainrate)
		}
		beta0Grad1 = beta0Grad
		
		gamma0Grad1 = gamma0Grad1 :* gamma0Grad
		sind = selectindex(gamma0Grad1[1,] :> 0)
		if (length(sind) > 0) {
			gamma0Gain[1,sind] = gamma0Gain[1,sind] :+ gainrate
		}
		sind = selectindex(gamma0Grad1[1,] :< 0)
		if (length(sind) > 0) {
			gamma0Gain[1,sind] = gamma0Gain[1,sind] :* (1-gainrate)
		}
		gamma0Grad1 = gamma0Grad
	}
}

void c_mlp2_gd::_update(`RV' batch_ind)
{
	_grad_gamma(batch_ind)
	_grad_beta()
	_grad_alpha(batch_ind)
	
	if (gainrate > 0) {
		_update_gain()
		
		alpha = alpha - lrate*(alphaGain :* alphaGrad)
		beta  = beta  - lrate*(betaGain  :* betaGrad)
		gamma = gamma - lrate*(gammaGain :* gammaGrad)

		if (!nobias) {
			alpha0 = alpha0 - lrate*(alpha0Gain :* alpha0Grad)
			beta0  = beta0  - lrate*(beta0Gain  :* beta0Grad)
			gamma0 = gamma0 - lrate*(gamma0Gain :* gamma0Grad)
		}
		return
	}

	alpha = alpha - lrate * alphaGrad
	beta  = beta  - lrate * betaGrad
	gamma = gamma - lrate * gammaGrad

	if (!nobias) {
		alpha0 = alpha0 - lrate * alpha0Grad
		beta0  = beta0  - lrate * beta0Grad
		gamma0 = gamma0 - lrate * gamma0Grad
	}
}

////////////////////////////////////////////////////////////////////////////////
// class c_mlp2_momentum

class c_mlp2_momentum extends c_mlp2_gd {
protected:
	`RM' alphaMom,  betaMom,  gammaMom
	`RM' alpha0Mom, beta0Mom, gamma0Mom
	
	void _init_extra()
	void _update()
public:
	void new()
}


void c_mlp2_momentum::new()
{
	lrate    = 0.01
	friction = 1
}

void c_mlp2_momentum::_init_extra()
{
	alphaMom = J(p,  m1,   0)
	betaMom  = J(m1, m2,   0)
	gammaMom = J(m2, mout, 0)
	if (!nobias) {
		alpha0Mom = J(1, m1,   0)
		beta0Mom  = J(1, m2,   0)
		gamma0Mom = J(1, mout, 0)
	}
}

void c_mlp2_momentum::_update(`RV' batch_ind)
{
	_grad_gamma(batch_ind)
	_grad_beta()
	_grad_alpha(batch_ind)

	friction = friction0*((iter-1)/iter)
	if (friction < 0) friction = 0

	alphaMom  = friction*alphaMom + lrate*alphaGrad
	betaMom   = friction*betaMom  + lrate*betaGrad
	gammaMom  = friction*gammaMom + lrate*gammaGrad
	if (!nobias) {
		alpha0Mom = friction*alpha0Mom + lrate*alpha0Grad
		beta0Mom  = friction*beta0Mom  + lrate*beta0Grad
		gamma0Mom = friction*gamma0Mom + lrate*gamma0Grad
	}

	alpha = alpha - alphaMom
	beta  = beta  - betaMom
	gamma = gamma - gammaMom
	if (!nobias) {
		alpha0 = alpha0 - alpha0Mom
		beta0  = beta0  - beta0Mom
		gamma0 = gamma0 - gamma0Mom
	}
}

////////////////////////////////////////////////////////////////////////////////
// class c_mlp2_nag

class c_mlp2_nag extends c_mlp2_momentum {
protected:
	void _update()
}

void c_mlp2_nag::_update(`RV' batch_ind)
{
	if (friction > 0) {
		alpha = alpha - friction*alphaMom
		beta  = beta  - friction*betaMom
		gamma = gamma - friction*gammaMom
		if (!nobias) {
			alpha0 = alpha0 - friction*alpha0Mom
			beta0  = beta0  - friction*beta0Mom
			gamma0 = gamma0 - friction*gamma0Mom
		}
	}

	_grad_gamma(batch_ind)
	_grad_beta()
	_grad_alpha(batch_ind)

	if (friction > 0) {
		// restore previous state
		alpha = alpha + friction*alphaMom
		beta  = beta  + friction*betaMom
		gamma = gamma + friction*gammaMom
		if (!nobias) {
			alpha0 = alpha0 + friction*alpha0Mom
			beta0  = beta0  + friction*beta0Mom
			gamma0 = gamma0 + friction*gamma0Mom
		}
	}

	friction = friction0*((iter-1)/(iter+2))
	if (friction < 0) friction = 0

	alphaMom  = friction*alphaMom + lrate*alphaGrad
	betaMom   = friction*betaMom  + lrate*betaGrad
	gammaMom  = friction*gammaMom + lrate*gammaGrad
	if (!nobias) {
		alpha0Mom  = friction*alpha0Mom + lrate*alpha0Grad
		beta0Mom   = friction*beta0Mom  + lrate*beta0Grad
		gamma0Mom  = friction*gamma0Mom + lrate*gamma0Grad
	}

	alpha = alpha - alphaMom
	beta  = beta  - betaMom
	gamma = gamma - gammaMom
	if (!nobias) {
		alpha0 = alpha0 - alpha0Mom
		beta0  = beta0  - beta0Mom
		gamma0 = gamma0 - gamma0Mom
	}
}

////////////////////////////////////////////////////////////////////////////////
// class c_mlp2_adagrad

class c_mlp2_adagrad extends c_mlp2_gd {
protected:
	`RS' epsilon
	`RM' alphaSq,  betaSq,  gammaSq
	`RM' alpha0Sq, beta0Sq, gamma0Sq
	
	void _init_extra()
	void _update()
public:
	void new()
	void set_eps()
}


void c_mlp2_adagrad::new()
{
	lrate    = 0.01
	epsilon  = 1e-8
}

void c_mlp2_adagrad::_init_extra()
{
	alphaSq = J(p,  m1,   0)
	betaSq  = J(m1, m2,   0)
	gammaSq = J(m2, mout, 0)
	if (!nobias) {
		alpha0Sq = J(1, m1,   0)
		beta0Sq  = J(1, m2,   0)
		gamma0Sq = J(1, mout, 0)
	}
}

void c_mlp2_adagrad::set_eps(`RS' eps)
{
	epsilon = eps
}

void c_mlp2_adagrad::_update(`RV' batch_ind)
{
	_grad_gamma(batch_ind)
	_grad_beta()
	_grad_alpha(batch_ind)

	alphaSq  = alphaSq + (alphaGrad:*alphaGrad)
	betaSq   = betaSq  + (betaGrad:*betaGrad)
	gammaSq  = gammaSq + (gammaGrad:*gammaGrad)
	if (!nobias) {
		alpha0Sq  = alpha0Sq + (alpha0Grad:*alpha0Grad)
		beta0Sq   = beta0Sq  + (beta0Grad:*beta0Grad)
		gamma0Sq  = gamma0Sq + (gamma0Grad:*gamma0Grad)
	}

	alpha = alpha - lrate*(alphaGrad :/ sqrt(alphaSq:+epsilon))
	beta  = beta  - lrate*(betaGrad  :/ sqrt(betaSq:+epsilon))
	gamma = gamma - lrate*(gammaGrad :/ sqrt(gammaSq:+epsilon))
	if (!nobias) {
		alpha0 = alpha0 - lrate*(alpha0Grad :/ sqrt(alpha0Sq:+epsilon))
		beta0  = beta0  - lrate*(beta0Grad  :/ sqrt(beta0Sq:+epsilon))
		gamma0 = gamma0 - lrate*(gamma0Grad :/ sqrt(gamma0Sq:+epsilon))
	}
}

end
