%% Fitting Exponential Model to 6-7 Meme Pageviews - Steepest Descent
% Joey Zhang - Optimization Project
% Trying to fit y(t) = A*exp(a*t) to the meme data
% Using steepest descent from Chapter 8
%
% Basic idea: take log of both sides to linearize it
% ln(y) = ln(A) + a*t  <-- this is just y = mx + b form!
% Then solve min ||Bx - b||^2 where x = [ln(A); a]

clear; close all; clc;

%% Setup / Config
data_file = '67_pageviews.txt';
n_fit = 10;         % use first 10 points for training
n_eval = 5;         % test on next 5 points
tol = 1e-7;         % stop when change is small enough
maxIter = 10000;    % safety limit so we don't run forever
print_every = 100;  % print updates every 100 iterations

%% Load the data from file
fprintf('Loading data from %s...\n', data_file);

% Open file and read it
fid = fopen(data_file, 'r');
if fid == -1
    error('Uh oh, cannot open file: %s', data_file);
end

% Read line by line, skip comments (lines starting with #)
data = [];
while ~feof(fid)
    line = fgetl(fid);
    if ischar(line) && ~isempty(line) && line(1) ~= '#'
        parts = strsplit(line, '\t');  % tab separated
        t_val = str2double(parts{1});
        views_val = str2double(parts{2});
        % date_str = parts{3};  % don't need date yet
        data = [data; t_val, views_val];
    end
end
fclose(fid);

fprintf('Loaded %d data points total.\n', size(data, 1));

%% Split into training and test sets
% Training: first 10 days
t_fit = data(1:n_fit, 1);
y_fit = data(1:n_fit, 2);

% Test: next 5 days
t_next = data(n_fit+1:n_fit+n_eval, 1);
y_next = data(n_fit+1:n_fit+n_eval, 2);

fprintf('Training set: %d points (t = %d to %d)\n', n_fit, t_fit(1), t_fit(end));
fprintf('Test set: %d points (t = %d to %d)\n', n_eval, t_next(1), t_next(end));

%% Build the linear system Bx = b
% Remember: we linearized by taking logs
% Original: y = A*exp(a*t)
% Log both sides: ln(y) = ln(A) + a*t
% 
% This is like: z = x1 + x2*t  where z = ln(y), x1 = ln(A), x2 = a
%
% In matrix form:  [1  t1] [x1]   [ln(y1)]
%                  [1  t2] [x2] = [ln(y2)]
%                  [...  ]        [ ...  ]
%                  [1  tn]        [ln(yn)]
%
% So B = [ones, t_fit] and b = log(y_fit)

B = [ones(n_fit, 1), t_fit];  % first column all 1s, second column is t
b = log(y_fit);                % take natural log of y values

fprintf('\nLinear system built:\n');
fprintf('  B is %d x %d\n', size(B, 1), size(B, 2));
fprintf('  b is %d x 1\n', size(b, 1));

%% Steepest Descent Algorithm
% This is from Chapter 8 of the textbook
% Goal: minimize f(x) = 0.5*||Bx - b||^2
% Gradient: grad_f = B^T*(Bx - b)
% Update: x_new = x - alpha * grad
% where alpha is step size (we'll use exact line search)

fprintf('\n========================================\n');
fprintf('Starting Steepest Descent\n');
fprintf('========================================\n');
fprintf('Stopping when ||x_new - x|| < %.1e\n', tol);
fprintf('Max iterations: %d\n\n', maxIter);

% Initialize x
% I'll use the normal equations solution as starting point
% x = (B^T*B)^(-1) * B^T * b
% (this should actually be optimal already but let's run SD anyway)
x = (B' * B) \ (B' * b);

fprintf('Starting point x0:\n');
fprintf('  x(1) = ln(A) = %.6f\n', x(1));
fprintf('  x(2) = a     = %.6f\n', x(2));
fprintf('  f(x0) = %.6e\n\n', 0.5 * norm(B*x - b)^2);

% Main loop
iter = 0;
converged = false;

while iter < maxIter
    % Step 1: compute gradient
    % grad = B^T*(Bx - b)  <-- from chain rule
    grad = B' * (B * x - b);
    
    % Step 2: descent direction is negative gradient
    g = grad;  % just renaming for clarity with textbook notation
    d = -g;    % steepest descent direction
    
    % Step 3: find optimal step size (exact line search)
    % For quadratic f(x) = 0.5*||Bx-b||^2, there's a formula!
    % Minimize f(x - alpha*g) over alpha > 0
    %
    % Taking derivative and setting = 0:
    % d/dalpha [0.5*||B(x - alpha*g) - b||^2] = 0
    %
    % Working through the math:
    % = d/dalpha [0.5*(Bx - alpha*Bg - b)^T*(Bx - alpha*Bg - b)]
    % = (Bx - alpha*Bg - b)^T * (-Bg)
    % = -(Bx - b)^T*Bg + alpha*(Bg)^T*Bg
    % = -g^T*g + alpha*||Bg||^2  (since g = B^T*(Bx-b))
    %
    % Set = 0 and solve:  alpha = (g^T*g) / ||Bg||^2
    %
    % This is equation 8.2 from the textbook!
    
    Bg = B * g;
    alpha = (g' * g) / (Bg' * Bg);  % exact formula for quadratics
    
    % Step 4: update x
    x_new = x + alpha * d;  % move in direction d with step size alpha
    
    % Step 5: check stopping criterion
    delta_x = norm(x_new - x);
    
    iter = iter + 1;
    
    % Print progress occasionally
    if mod(iter, print_every) == 0 || iter == 1
        f_val = 0.5 * norm(B*x - b)^2;
        fprintf('Iter %5d: f = %.6e, ||grad|| = %.6e, ||delta_x|| = %.6e, alpha = %.6e\n', ...
                iter, f_val, norm(grad), delta_x, alpha);
    end
    
    % Did we converge?
    if delta_x < tol
        converged = true;
        fprintf('\n>>> Converged at iteration %d! <<<\n', iter);
        fprintf('    ||x_new - x|| = %.6e < %.6e\n', delta_x, tol);
        break;
    end
    
    % Update for next iteration
    x = x_new;
end

if ~converged
    warning('Hit max iterations (%d) without converging!', maxIter);
end

%% Extract the fitted model parameters
fprintf('\n========================================\n');
fprintf('Final Results\n');
fprintf('========================================\n');

% Remember: x = [ln(A); a], so need to take exp to get A
ln_A = x(1);
a_param = x(2);
A_param = exp(ln_A);  % undo the log transformation

fprintf('Fitted parameters:\n');
fprintf('  ln(A) = %.8f\n', ln_A);
fprintf('  a     = %.8f\n', a_param);
fprintf('  A     = exp(ln(A)) = %.4f\n', A_param);
fprintf('\nFitted exponential model:\n');
fprintf('  y(t) = %.4f * exp(%.6f * t)\n', A_param, a_param);

% What's the final objective value?
f_final = 0.5 * norm(B*x - b)^2;
fprintf('\nFinal objective f(x) = %.6e\n', f_final);

%% Make predictions for the next 5 points
fprintf('\n========================================\n');
fprintf('Predictions on Test Set\n');
fprintf('========================================\n');

% Use fitted model: y(t) = A * exp(a*t)
y_pred = A_param * exp(a_param * t_next);

% Need to get dates again for nice output
fid = fopen(data_file, 'r');
dates = cell(15, 1);
idx = 1;
while ~feof(fid)
    line = fgetl(fid);
    if ischar(line) && ~isempty(line) && line(1) ~= '#'
        parts = strsplit(line, '\t');
        dates{idx} = parts{3};
        idx = idx + 1;
    end
end
fclose(fid);

% Print results table
fprintf('\n');
fprintf('  t  |    Date    | Predicted | Actual  | Error (%%)\n');
fprintf('-----|------------|-----------|---------|----------\n');
for i = 1:n_eval
    t_val = t_next(i);
    date_str = dates{n_fit + i};
    pred_val = y_pred(i);
    actual_val = y_next(i);
    error_pct = 100 * abs(pred_val - actual_val) / actual_val;
    fprintf(' %2d  | %s | %9.1f | %7.0f | %6.2f%%\n', ...
            t_val, date_str, pred_val, actual_val, error_pct);
end
fprintf('\n');
fprintf('Note: Predictions get worse because the meme peaked and started declining,\n');
fprintf('      but exponential model only knows how to grow!\n\n');

%% Make a nice plot
fprintf('========================================\n');
fprintf('Creating visualization...\n');
fprintf('========================================\n');

figure('Position', [100, 100, 900, 600]);
hold on;

% Plot all the actual data
t_all = data(:, 1);
y_all = data(:, 2);
plot(t_all, y_all, 'ko', 'MarkerSize', 8, 'LineWidth', 1.5, ...
     'DisplayName', 'Actual Data (all 15 points)');

% Plot the fitted exponential curve
t_smooth = linspace(0, 14, 200);  % smooth curve
y_fitted = A_param * exp(a_param * t_smooth);
plot(t_smooth, y_fitted, 'b-', 'LineWidth', 2, ...
     'DisplayName', sprintf('Fitted: y(t) = %.1f exp(%.4f t)', A_param, a_param));

% Highlight training data
plot(t_fit, y_fit, 'bs', 'MarkerSize', 10, 'LineWidth', 2, ...
     'DisplayName', 'Training Data (days 0-9)');

% Show predictions vs actual for test set
plot(t_next, y_pred, 'r^', 'MarkerSize', 12, 'LineWidth', 2, ...
     'MarkerFaceColor', 'r', 'DisplayName', 'Predictions (days 10-14)');
plot(t_next, y_next, 'gd', 'MarkerSize', 12, 'LineWidth', 2, ...
     'MarkerFaceColor', 'g', 'DisplayName', 'Actual (days 10-14)');

% Labels and formatting
grid on;
xlabel('Time t (days since Dec 13, 2025)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Wikipedia Pageviews', 'FontSize', 12, 'FontWeight', 'bold');
title('Exponential Fit to 6-7 Meme Pageviews (Steepest Descent)', ...
      'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 10);
set(gca, 'FontSize', 11);
set(gca, 'GridAlpha', 0.3);

% Make sure everything fits nicely
ylim([0, max(max(y_all), max(y_pred)) * 1.1]);

hold off;

% Save it
output_file = '67_fit_plot.png';
saveas(gcf, output_file);
fprintf('Plot saved as: %s\n', output_file);
fprintf('\nDone!\n');
