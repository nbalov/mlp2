*! version 1.0.0  22nov2017

include mlp2.mata, adopath

////////////////////////////////////////////////////////////////////////////////

program mlp2, eclass

	gettoken subcmd lhs : 0

	if `"`subcmd'"' == "fit" {
		mlp2_fit `lhs'
	}
	else if `"`subcmd'"' == "predict" {
		mlp2_predict `lhs'
	}
	else if `"`subcmd'"' == "simulate" {
		mlp2_simulate `lhs'
	}
	else {
		di as err "mlp2: unknown command {bf:`subcmd'}"
		exit 198
	}
end

program mlp2_fit, eclass

	syntax varlist(fv ts) [if] [in] [, NOINTERcepts	///
		layer1(real 0) layer2(real 0)		///
		OPTimizer(string)			///
		LRate(real 0.01) GAINRate(real 0)	///
		FRiction(real 1)			///
		EPSilon(real 1e-8)			///
		LOSStol(real 0.00)			///
		DROPout(real 0.00)			///
		epochs(real 1000)      			///
		batch(real 50)				///
		echo(real 0)]

	tempvar touse
	marksample touse
	markout `touse' `varlist'

	gettoken y varlist : varlist
	local xvars `varlist'
        _fv_check_depvar `y'

	_labels2eqnames `y'
        local eqlist0 `"`r(eqlist)'"'
	local nout = r(k_eq)
	local ylabels = `"`r(labels)'"'

	tempvar yy
	egen `yy' = group(`y')

	if `layer1' <= 0 {
		local layer1 = `nout'
	}
	
	if `layer2' <= 0 {
		local layer2 = `layer1'
	}
	
	if `lrate' <= 0 {
		di as err "{\bf lrate()} must be positive"
		error 198
	}

	if `gainrate' < 0 {
		di as err "{\bf gainrate()} must be non-negative"
		error 198
	}
	
	if `friction' < 0 | `friction' > 1 {
		di as err "{\bf friction()} must be between 0 and 1"
		error 198
	}

	if `epsilon' <= 0 {
		di as err "{\bf epsilon()} must be positive"
		error 198
	}
	
	if `epochs' < 1 {
		di as err "{\bf epochs()} must be positive integer"
		error 198
	}
	
	if `losstol' < 0 {
		di as err "{\bf losstol()} must be positive"
		error 198
	}
	
	if `dropout' < 0 | `dropout' >= 1 {
		di as err "{\bf dropout()} must be a probability between 0 and 1"
		error 198
	}

	local rstate = c(rngstate)

	if `"`optimizer'"' == "" {
		local optimizer gd
	}
	if `"`optimizer'"' == "gd" {
		local c_mlp2_optimizer c_mlp2_gd
	}
	else if `"`optimizer'"' == "momentum" || `"`optimizer'"' == "mom" {
		local c_mlp2_optimizer c_mlp2_momentum
	}
	else if `"`optimizer'"' == "nag" {
		local c_mlp2_optimizer c_mlp2_nag
	}
	else if `"`optimizer'"' == "adagrad" {
		local c_mlp2_optimizer c_mlp2_adagrad
	}
	else {
		di as err "unknown optimizer {bf:`optimizer'}"
		error 198
	}
	
	mata: _mlp2 = `c_mlp2_optimizer'()

	mata: _mlp2.set_lrate(`lrate')
	mata: _mlp2.set_gainrate(`gainrate')
	mata: _mlp2.set_friction(`friction')
	if `"`optimizer'"' == "momentum" {
	}
	else if `"`optimizer'"' == "nag" {
	}
	else if `"`optimizer'"' == "adagrad" {
		mata: _mlp2.set_eps(`epsilon')
	}

	mata: _mlp2.fit(`"`yy'"', `"`xvars'"',	`"`touse'"',	///
		`layer1', `layer2', `nout',		 	///
		`batch', `epochs', `losstol', `dropout',	///
		 "`nointercepts'" != "", `echo')
	mata: _mlp2.save_params()
	mata: mata drop _mlp2

	ereturn hidden local yvar    = `"`y'"'
	ereturn hidden local xvars   = `"`xvars'"'
	ereturn hidden local ylabels = `"`ylabels'"'
	ereturn hidden local rstate  = `"`rstate'"'
end

program mlp2_predict, eclass

	syntax [anything] [if] [in] [, genvar(string) truelabel(string)]
	if `"`anything'"' != "" {
		local 0 `"`anything'"'
		syntax [varlist(fv ts)]
	}
	else {
		local varlist `e(xvars)'
	}

	if `"`truelabel'"' == "" {
		local truelabel `e(yvar)'
	}
	
	if `"`genvar'"' == "" {
		local genvar predict
	}

	tempvar touse
	marksample touse
	markout `touse' `varlist' `truelabel'

	mata: _mlp2 = c_mlp2()
	mata: _mlp2.predict(`"`varlist'"', `"`touse'"', `"`genvar'"')
	if `"`truelabel'"' != "" {
		tempvar yy
		egen `yy' = group(`truelabel')
		mata: _mlp2.score(`"`yy'"', `"`touse'"')
		di 
		di "Prediction accuracy: `e(score_acc)'"
	}
	mata: mata drop _mlp2
end

program mlp2_simulate, eclass

	syntax varlist(fv ts) [if] [in] [, genvar(string)]

	if `"`genvar'"' == "" {
		local genvar simulate
	}

	tempvar touse
	marksample touse
	markout `touse' `varlist'
	
	mata: _mlp2 = c_mlp2()
	mata: _mlp2.simulate(`"`varlist'"', `"`touse'"', `"`genvar'"')
	mata: mata drop _mlp2
end

////////////////////////////////////////////////////////////////////////////////
