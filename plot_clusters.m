function [] = plot_clusters(mu, cov, col, varargin)

if nargin < 4
    varargin = {'LineStyle','-'};
end

if nargin < 3
    col = 'k';
end

% Code written by Lawrence Bull
k = size(mu, 1);
% hold on
% plot the ellipses
theta = 0:0.1:2 * pi;
for i = 1:k
    [vec, val] = eig(cov(:,:,i));
    alph = atan(vec(2,1)/vec(1,1));
    
    for j = 2:2
        cx = j*val(1,1)^0.5*cos(theta);
        cy = j*val(2,2)^0.5*sin(theta);
        cr = [cos(alph), -sin(alph); sin(alph), cos(alph)]*[cx; cy];
        c = cr + mu(i,:)'; 
        plot(c(1,:), c(2,:), 'Color', col, varargin{:})
    end
end
% plot the mean
if length(col) == 4
    scatter(mu(:,1), mu(:,2), 80, col(1:3), '+', 'LineWidth', 2, 'MarkerFaceAlpha',col(4),'MarkerEdgeAlpha',col(4))
else
    scatter(mu(:,1), mu(:,2), 80, col, '+', 'LineWidth', 2)
end
% hold off
end