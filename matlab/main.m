clear
clc

% 注意，matlab当中默认进行列操作
load('../dat/data_trajectory_1.txt');
data_block = data_trajectory_1;

min_value = min(data_block, [], "all");
max_value = max(data_block, [], "all");
data_block = (data_block-min_value)./(max_value-min_value);

rng(42);
k = 5;
PComponents = zeros(1, k);
PComponents(:) = 1/5;
[idx, mu] = kmeans(data_block, k, 'Start', 'plus', 'MaxIter', 100, 'Display', 'iter');

for i = 1:k
    cluster_points = data_block(idx == i, :);  % 获取属于第 i 个簇的所有点
    cov_matrix{i} = cov(cluster_points);    % 计算该簇的协方差矩阵
    fprintf('Covariance matrix for cluster %d:\n', i);
    disp(cov_matrix{i});  % 显示协方差矩阵
end

sigma = zeros(size(cov_matrix{1}));
for nc = 1:numel(cov_matrix)
    sigma = sigma + 1/k*cov_matrix{nc};
end

%%
para_init = struct('mu', mu, 'Sigma', sigma, 'ComponentProportion', PComponents);
% GMModel = fitgmdist(data_block, k, "CovarianceType", "full", "SharedCovariance", true, ...
%     "Start", para_init);
GMModel = fitgmdist(data_block, k, "CovarianceType", "full", "SharedCovariance", true, ...
    "Start", "plus");