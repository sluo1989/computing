function [V, W] = nnegmf(X, r, varargin)
% NNEGMF Non-negative matrix factorization by multiplicative updates
%   [V, W] = NNEGMF(X, r) finds m-by-r V and r-by-n W with nonnegative
%   entries that such that norm(X - V * W, 'fro') is minimal.
%
% INPUT:
%   X - m-by-n data matrix with nonnegative entries
%   r - a positive rank
%
% OPTIONAL NAME-VALUE PAIRS:
%   'device' - device for computation; 'cpu' (default) or 'gpu'
%   'maxiter' - maximal number of iterations
%   'precision' - precision for GPU; 'single' (default) or 'double'
%   'tolfun' - convergence tolerance for relative change in objective value
%   'V0' - starting point for V
%   'W0' - starting point for W
%
% OUTPUT:
%   V, W - optimal V and W
%
% See also nnmf.
%
% Example
%
% References
%

% Copyright 2015 North Carolina State University
% Hua Zhou (hua_zhou@ncsu.edu)

% input parsing rule
[m, n] = size(X);
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('r', @(x) isnumeric(x) && r > 0);
argin.addParameter('device', 'cpu', @ischar);
argin.addParameter('maxiter', 1e4, @isnumeric);
argin.addParameter('precision', 'single', @ischar);
argin.addParameter('tolfun', 1e-4, @isnumeric);
argin.addParameter('V0', rand(m, r), @isnumeric);
argin.addParameter('W0', rand(r, n), @isnumeric);

% parse inputs
argin.parse(X, r, varargin{:});
device = argin.Results.device;
maxiter = argin.Results.maxiter;
precision = argin.Results.precision;
tolfun = argin.Results.tolfun;
V = max(argin.Results.V0, 1e-8); % stay away from sticky boundary
W = max(argin.Results.W0, 1e-8); % stay away from sticky boundary

if strcmpi(device, 'cpu')
    
    % MM loop
    B = V * W;
    obj = norm(X - B, 'fro')^2;
    for iter = 1:maxiter
        % multiplicative update of V and W
        V = V .* (X * W') ./ (B * W');
        B = V * W;
        W = W .* (V' * X) ./ (V' * B);
        B = V * W;
        % check stopping criterion
        objold = obj;
        obj = norm(X - B, 'fro')^2;
        if abs(obj - objold) < tolfun * (objold + 1)
            break
        end
    end
    
elseif strcmpi(device, 'gpu')
    
    % transfer data to GPU
    if strcmpi(precision, 'single')
        X_g = gpuArray(single(X));
        V_g = gpuArray(single(V));
        W_g = gpuArray(single(W));
    elseif strcmpi(precision, 'double')
        X_g = gpuArray(X);
        V_g = gpuArray(V);
        W_g = gpuArray(W);
    end
    
    % MM loop
    B_g = V_g * W_g;
    obj = norm(X_g - B_g, 'fro')^2;
    for iter = 1:maxiter
        % multiplicative update of V and W
        V_g = V_g .* (X_g * W_g') ./ (B_g * W_g');
        B_g = V_g * W_g;
        W_g = W_g .* (V_g' * X_g) ./ (V_g' * B_g);
        B_g = V_g * W_g;
        % check stopping criterion
        objold = obj;
        obj = norm(X_g - B_g, 'fro')^2;
        if abs(obj - objold) < tolfun * (objold + 1)
            break
        end
    end
    
    % retrieve result from GPU
    V = gather(V_g);
    W = gather(W_g);
    
end

end
