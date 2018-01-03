{smcl}
{* *! version 1.0.0  22nov2017}{...}
{vieweralsosee "" "--"}{...}
{viewerjumpto "Syntax" "mlp2##syntax"}{...}
{viewerjumpto "Menu" "mlp##menu"}{...}
{viewerjumpto "Description" "mlp##description"}{...}
{viewerjumpto "Options" "mlp##options"}{...}
{viewerjumpto "Remarks" "mlp##remarks"}{...}
{viewerjumpto "Examples" "mlp##examples"}{...}
{viewerjumpto "Stored results" "mlp##results"}{...}
{viewerjumpto "Author" "mlp##author"}{...}
{title:Title}

{p2colset 5 15 17 2}{...}
{p2col :{cmd:mlp2} {hline 2}}Multilayer perceptron with 2 hidden layers{p_end}
{p2colreset}{...}


{marker syntax}{...}
{title:Syntax}

{phang}
Training a model

{p 8 11 2}
{opt mlp2 fit} {depvar} [{indepvars}] {ifin} [{cmd:,} {it:fit_options}]

{phang}
{depvar} is a categorical variable with unordered outcome levels.

{synoptset 25 tabbed}{...}
{marker fit_options_table}{...}
{synopthdr}
{synoptline}
{synopt :{opt layer1(#)}}numbers of neurons in the 1-st hidden layer; 
default is the number of levels of {depvar}{p_end}
{synopt :{opt layer2(#)}}numbers of neurons in the 2-nd hidden layer; 
default is {opt level1}{p_end}
{synopt :{opt nobias}}no bias terms are used{p_end}
{synopt :{opt opt:imizer(string)}}optimizer; default is {cmd:optimizer(gd)}{p_end}
{synopt :{opt lr:ate(#)}}learning rate of the optimizer; default is {cmd:lrate(0.01)}{p_end}
{synopt :{opt fr:iction(#)}}friction rate for momentum optimizers; default is {cmd:friction(1)}{p_end}
{synopt :{opt eps:ilon(#)}}gradient smoothing term; default is {cmd:epsilon(1e-8)}{p_end}
{synopt :{opt loss:tol(#)}}stopping loss tolerance; default is {cmd:losstol(0)}{p_end}
{synopt :{opt drop:out(#)}}dropout probability; default is {cmd:dropout(0)}{p_end}
{synopt :{opt batch(#)}}training batch size; default is {cmd:batch(50)} or entire sample{p_end}
{synopt :{opt epochs(#)}}maximum number of iterations; default is {cmd:epochs(1000)}{p_end}
{synopt :{opt echo(#)}}report loss values at every # number of iterations;
 defailt is {cmd:echo(0)}{p_end}
{synoptline}

{phang}
Prediction using already trained model

{p 8 11 2}
{opt mlp2 predict} [{indepvars}] {ifin} [{cmd:,} {it:predict_options}]

{phang}
{indepvars} specified in {opt mlp2 fit} and {opt mlp2 predict} should be compatible.

{synoptset 25 tabbed}{...}
{marker predict_options_table}{...}
{synopthdr}
{synoptline}
{synopt :{opt genvar(newname)}}stub used for prediction variables;
default is {cmd:genvar(predict)}{p_end}
{synopt :{opt truelabel(varname)}}name of the variable with true labels; 
default is {depvar} used in {opt mlp2 fit}{p_end}
{synoptline}


{marker description}{...}
{title:Description}

{pstd}
{cmd:mlp2 fit} trains a multilayer perceptron with 2 hidden layers using the 
current dataset. 

{pstd}
{cmd:mlp2 predict} makes a prediction based on a previously trained network 
using {cmd:mlp2 fit}. 

{marker options}{...}
{title:Options}

{phang}
{opt layer1(#)} specifies the numbers of neurons in the 1-st hidden layer. 
Default is the number of levels of the outcome variable.

{phang}
{opt layer2(#)} specifies the numbers of neurons in the 2-nd hidden layer. 
By default, it equals the number of neurons in the 1-st layer.

{phang}
{opt nobias} requests that no bias terms are used at each level of the network. 
By default, each neuron has a bias term added to the linear combination of the 
inbound neurons.

{phang}
{opt optimizer(string)} specifies a gradient based optimizer algorithm used for 
training. Available options are {it: gd}, a regular gradient descent, 
{it: momentum}, a momentum optimization algorithm, {it: nag}, 
Nesterov accelerated gradient algorithm, and {it: adagrad}, the AdaGrad algorithm. 
By default {it: gd} is used. 

{phang}
{opt lrate(#)} specifies the learning rate used by gradient descent algorithms. 
It must be a positive number. The default value is {it:0.01}.

{phang}
{opt friction(#)} specifies the friction rate for momentum based optimizers. 
It must be a number between 0 and 1, including. The default value is {it:1}.

{phang}
{opt epsilon(#)} specifies the gradient smoothing term used in AdaGrad and similar 
optimization algorithms. It must be a positive number. The default value is {it:1e-8}.

{phang}
{opt losstol(#)} specifies the stopping loss tolerance used by all 
optimization algorithms. It must be a non-negative number. The default value is {it:0}.

{phang}
{opt dropout(#)} specifies the dropout probability applied during training. 
It must be a probability value less than 1. The default value is {it:0}.

{phang}
{opt batch(#)} specifies the batch size used during training. 
By default, batch of size 50 is used, {opt batch(50)}. If you want to use the 
entire sample as one batch, you specify {opt batch(0)}. 

{phang}
{opt epochs(#)} specifies the maximum number of iterations performed by the 
optimizer. The default value is {it:1000}.

{phang}
{opt echo(#)} requests that the loss is reported at every # number of 
iterations during training. By default the loss is not reported, {opt echo(0)}.

{phang}
{opt genvar(newname)} specifies a stub used for generating variables that contain 
prediction probabilities. For example if {depvar} has two levels 
{it:yes} and {it:no}, then the {cmd:mlp2 predict} command will generate two 
variables named {it:newname_yes} and {it:newname_no} with probabilities that 
sum to 1.

{phang}
{opt truelabel(varname)} specifies a variable with the true labels of the sample. 
It is used for validating the accuracy of prediction. 
By default this is {depvar} used during training. 

{marker remarks}{...}
{title:Remarks}

{pstd}
{cmd:mlp2} implements a multilayer perceptron with 2 hidden layers where 
the neurons at each layer are fully connected to the neurons of the lower layer. 
The variables {indepvars} define the neurons of the lowest, zero, layer. 
The highest, third, layer has as many neurons as is the number of levels of the 
outcome variable {depvar}. 

{pstd}
The activation functions used for the two hidden 
layers are rectified liner units, {cmd:x -> max(0, x)}. The activation function 
in the third, output, layer is the softmax function. That is, conditional on 
the second hidden layer, the third layer implements multinomial logistic 
regression. The loss function of the network is given by the negative 
log-likelihood of the multinomial logistic regression model.

{pstd}
The training command {cmd:mlp2 fit} uses stochastic gradient descent algorithms 
to optimize the loss function. You can control it by specifying appropriate 
batch size, {opt batch()}, learning rate, {opt lrate()}, etc.

{marker examples}{...}
{title:Examples}

{pstd}Training and prediction evaluation of the MNIST dataset{p_end}

{phang2}{cmd:. use mnist-train}{p_end}
{phang2}{cmd:. mlp2 fit y v*, hidden1(100) hidden2(100) epochs(100)}{p_end}
{phang2}{cmd:. mlp2 predict, genvar(ypred) truelabel(y)}{p_end}


{marker results}{...}
{title:Stored results}

{pstd}
{cmd:mlp2} stores the following in {cmd:e()}:

{synoptset 15 tabbed}{...}

{p2col 5 15 17 2: Matrices}{p_end}
{synopt:{cmd:e(alpha)}}Matrix of alpha coefficients{p_end}
{synopt:{cmd:e(beta)}}Matrix of beta coefficients{p_end}
{synopt:{cmd:e(gamma)}}Matrix of gamma coefficients{p_end}

{synopt:{cmd:e(alpha0)}}Matrix of alpha intercepts{p_end}
{synopt:{cmd:e(beta0)}}Matrix of beta intercepts{p_end}
{synopt:{cmd:e(gamma0)}}Matrix of gamma intercepts{p_end}

{marker Author}{...}
{title:Author}

{pstd}
Nikolay Balov, nbalov@stata.com
