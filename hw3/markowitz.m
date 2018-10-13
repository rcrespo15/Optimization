%% CE 295 - Energy Systems and Control
%   Quadratic Programming Example
%   Markowitz Portfolio Optimization

clear; close all;
fs = 18;    % Font Size for plots

%% Problem Data
mu = [1.02; 1.05; 1.04];
Sig = diag([0.05^2, 0.5^2, 0.1^1]);

N = 21;
gam = linspace(0,1,N);

%% Solve QP
% Preallocate
x_star = nan*ones(3,N);
expected_return = nan*ones(N,1);
risk = nan*ones(N,1);

for idx = 1:N
    
    % CVX
    cvx_begin
        variable x(3);
        minimize( -mu'*x + gam(idx)*x'*Sig*x );
        subject to
            ones(3,1)'*x == 1;
            x >= 0;
    cvx_end

    x_star(:,idx) = x;
    expected_return(idx) = mu'*x;
    risk(idx) = x'*Sig*x;

end

%% Plot Results

figure(1); clf;
plot(expected_return,risk,'LineWidth',2)
xlabel('Expected Return [%]');
ylabel('Risk')
set(gca,'FontSize',fs)

figure(2); clf;
bar(gam,x_star','stacked')
legend('x_1','x_2','x_3','Location','SouthEast')
xlabel('Risk Aversion Parameter, \gamma')
ylabel('Portfolio Distribution')
set(gca,'FontSize',fs)
xlim([-0.05,1.05])

%% SOCP w/ Chance Constraint on Limited Loss
% Parameters
eta_vec = 0.90:0.01:0.99
N = length(eta_vec);

% Preallocate
x_star = nan*ones(3,N);
expected_return = nan*ones(N,1);

for idx = 1:length(eta_vec)
    
    % CVX
    cvx_begin
        variable x(3);
        minimize( -mu'*x );
        subject to
            x >= 0
            sum(x) == 1;
            norminv(eta_vec(idx),0,1)*norm(Sig^0.5*x,2) <= mu'*x - 0.9;
    cvx_end

    x_star(:,idx) = x;
    expected_return(idx) = mu'*x;

end

%% Plot Results

figure(3); clf;
bar(eta_vec,x_star','stacked')
legend('x_1','x_2','x_3','Location','SouthEast')
xlabel('Reliability, \eta')
ylabel('Portfolio Distribution')
set(gca,'FontSize',fs)
xlim([0.89,1.00])