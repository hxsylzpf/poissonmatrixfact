%CODE FOR POISSON MATRIX FACTORIZATION ON 20NEWSGROUP DATASET
clear;

% load the data (documents x words matrix, N documents, M words)
load('20newsgroups');
[N M] = size(X);

% hyperparameters of the gamma priors on U,V,Lambda 
a_u = 1;b_u = 1e-6;
a_v = 1;b_v = 1e-6;

% number of latent factors
K = 20;

% initialization
U = gamrnd(a_u,1/b_u,N,K); % NxK matrix, drawn randomly from the prior
V = gamrnd(a_v,1/b_v,M,K); % MxK matrix, drawn randomly from the prior

% max_iters = 1000;
% burnin = 500;
max_iters = 1000;
burnin = 500;

% prepare the training and test data

[I J vals] = find(X); % all nonzero entries (with indices) in X
% vals is a vector containing denotes the nonzero entries in X
% I and J are vectors with row and column indices resepectively

% Do an 80/20 split of matrix entries (nonzeros) into training and test data
num_elem = length(I);
rp = randperm(num_elem);

% indices of training set entries
I_train = I(rp(1:round(0.8*num_elem)));
J_train = J(rp(1:round(0.8*num_elem)));
Z = length(I_train);
% indices of test set entries
J_test = J(rp(round(0.8*num_elem)+1:end));
I_test = I(rp(round(0.8*num_elem)+1:end));
Y = length(I_test);
% values of the training and test entries
vals_train = vals(rp(1:round(0.8*num_elem)));
vals_test = vals(rp(round(0.8*num_elem)+1:end));

% run the Gibbs sampler for max_iters iterations
% in each iteration, draw samples of U, V, lambda
ex = zeros(Z,K);
dummy = zeros(Z,1);
dummy2 = zeros(Y,1);
for iters=1:max_iters
    %Sampling X_nmk from multinomial
    for i = 1:Z
        n=I_train(i);
        m=J_train(i);
        p = (U(n,:)).*(V(m,:));
        p=p/sum(p);
        ex(i,:) = mnrnd(vals_train(i),p);
    end
    S = zeros(N,K);
    T = zeros(M,K);
    for i = 1:Z
        n=I_train(i);
        m=J_train(i);
        S(n,:)= S(n,:) + ex(i,:);
        T(m,:)= T(m,:) +ex(i,:);
    end
    S = S + a_u;
    T = T + a_v;

    % Sample U using its local conditional posterior
    theta =  sum(V) + b_u;
    for j = 1:N
        U(j,:) = gamrnd(S(j,:),1.0./theta);
    end
    % Sample V using its local conditional posterior
    theta =  sum(U) + b_v;
    for j = 1:M
        V(j,:) = gamrnd(T(j,:),1.0./theta);
    end
    
    % Approach 1 (using current samples of U and V from this iteration) 
    mae_train = 0;
    mae_test = 0;
    for i = 1:Z
        n=I_train(i);
        m=J_train(i);
        mae_train = mae_train + abs(vals_train(i) - U(n,:)*V(m,:)');
    end
    for i = 1:Y
        n=I_test(i);
        m=J_test(i);
        mae_test = mae_test + abs(vals_test(i) - U(n,:)*V(m,:)');
    end
    mae_train = mae_train / Z;
    mae_test = mae_test / Y;
    fprintf('Done with iteration %d, MAE_train = %f, MAE_test = %f\n',iters,mae_train,mae_test);
    
    % Approach 2 (using Monte Carlo averaging; but only using the
    % post-burnin samples of U and V)
    if iters > burnin
        mae_train_avg = 0;
        mae_test_avg = 0;
        num = iters - burnin;
        for i=1:Z           
            n=I_train(i);
            m=J_train(i);
            %Maintains running sum of X_nm
            dummy(i)= dummy(i)+(U(n,:)*V(m,:)');
            pred = dummy(i)/num;
            mae_train_avg = mae_train_avg + abs(vals_train(i)-pred);
        end
        for i=1:Y           
            n=I_test(i);
            m=J_test(i);
            %Maintains running sum of X_nm
            dummy2(i)= dummy2(i)+(U(n,:)*V(m,:)');
            pred = dummy2(i)/num;
            mae_test_avg = mae_test_avg + abs(vals_test(i)-pred);
        end      
         mae_train_avg = mae_train_avg/Z;
         mae_test_avg = mae_test_avg/Y;
         fprintf('With Posterior Averaging, MAE_train_avg = %f, MAE_test_avg = %f\n',mae_train_avg,mae_test_avg);
    end
end
% Print the K topics (top 20 words from each topic)
% Take the V matrix and finds the 20 largest entries in
% each column of V. 
%call printtopics function to print the topics
printtopics(V);
