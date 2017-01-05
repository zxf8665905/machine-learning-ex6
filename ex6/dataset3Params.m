function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0;
sigma = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_values=[ 0.01,0.03,0.1,0.3,1,3,10,30];
sigma_values=[ 0.01,0.03,0.1,0.3,1,3,10,30];
error=zeros(length(C_values),length(sigma_values));
for i=1:length(C_values)
    for j=1:length(sigma_values)
        model= svmTrain(X, y, C_values(i), @(x1, x2) gaussianKernel(x1, x2, sigma_values(j)));
        pred=svmPredict(model,Xval);
        error(i,j)=mean(double(pred~=yval));
    end
end

[I_row, I_col] = find(min(min(error))==error);
C=C_values(I_row(1));
sigma=sigma_values(I_col(1));


% =========================================================================

end
