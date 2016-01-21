clear;
X = dlmread('nnmf-2429-by-361-face.txt');
V0full = dlmread('V0.txt');
W0full = dlmread('W0.txt');

rlist = 10:10:50;
runtime = zeros(length(rlist), 3);
objval = zeros(length(rlist), 3);
for i = 1:length(rlist)
    % rank
    r = rlist(i);
    display(['rank = ' num2str(r)]);
    % starting value
    V0 = V0full(:, 1:r);
    W0 = W0full(1:r, :);
    % CPU
    tic;
    [V1, W1] = nnegmf(X, r, 'V0', V0, 'W0', W0);
    runtime(i, 1) = toc;
    objval(i, 1) = norm(X - V1 * W1, 'fro')^2;
    % GPU SP
    tic;
    [V2, W2] = nnegmf(X, r, 'V0', V0, 'W0', W0, 'device', 'gpu');
    runtime(i, 2) = toc;
    objval(i, 2) = norm(X - V2 * W2, 'fro')^2;
    % GPU DP
    tic;
    [V3, W3] = nnegmf(X, r, 'V0', V0, 'W0', W0, 'device', 'gpu', ...
        'precision', 'double');
    runtime(i, 3) = toc;
    objval(i, 3) = norm(X - V3 * W3, 'fro')^2;
end

table(rlist', runtime(:, 1), runtime(:, 2), runtime(:, 3), ...
    objval(:, 1) / 1e3, objval(:, 2) / 1e3, objval(:, 3) / 1e3, ...
    'VariableNames', {'Rank', 'CPU_Time', ...
    'GPU_SP_Time', 'GPU_DP_Time', 'CPU_Obj', 'GPU_SP_Obj', 'GPU_DP_Obj'})

r = 25;
V0 = V0full(:, 1:r);
W0 = W0full(1:r, :);
[V, W] = nnegmf(X, r, 'V0', V0, 'W0', W0);

% display 25 basis images
figure; hold on;
set(gca, 'FontSize', 20);
for i = 1:25
    subplot(5, 5, i);
    imagesc(reshape(W(i, :), 19, 19));
    axis equal;
    axis tight;
end
