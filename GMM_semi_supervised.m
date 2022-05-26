classdef GMM_semi_supervised
    % Ignore property warning
    %#ok<*PROPLC>
    %#ok<*PROP>
    
    properties
        
        opts = struct();
        prior
        cluster
        D
        X
        label
        responsibility
        debug = false;
        fixed
        
    end
    
    properties (Dependent)
        nclust
        pi % Mixing proportions
        N
        Nx
    end
    
    methods
        
        function self = GMM_semi_supervised(X, y, prior, opts)
            
            if nargin > 3
                self.opts = opts;
            end
            
            if nargin > 2
                self.prior = prior;
            else
                error('Must Supply Prior')
            end
            
            
            nclust = length(self.prior.alpha);
            self.cluster = {};
            self.D = size(X,2);
            self.X = X;
            self.label = y;
            self.fixed = y>0;
            
            
            for nn = 1:nclust
                
                self.cluster{nn} = NIW(...
                    self.prior.m0,...
                    self.prior.k0,...
                    self.prior.n0,...
                    self.prior.S0,...
                    X(y==nn,:)...
                    );
                
            end
            
        end
        
        function self = add_data(self,X,y)
            
            self.X = [self.X;X];
            self.label = [self.label;y];
            
            % Assign labelled data
            for nn = 1:self.nclust
               
                self.cluster{nn} = self.cluster{nn}.add_data(X(y==nn,:));
                
            end
            
            % Assign unlabelled data to MAP
            inds = find(self.label == 0);
            for ii = 1:length(inds)
                
                x = X(inds(ii),:);
                % Normalised posterior loglikelihoods
                ll = self.nLL(self.predict_posterior(x));
                % Greedy cluster
                [~,nn] = max(exp(ll));
                
                % Add to cluster update labels
                self.cluster{nn} = self.cluster{nn}.add_one(x);
                self.label(inds(ii)) = nn;
                
            end
            
            
            
        end
        
        function self = rem_data(self,X,y) %#ok<INUSD>
            error('Who wants to remove data from their model anyway? Weirdo.')
        end
        
        function self = initialise(self)
            
            if isfield(self.opts,'initialisation_strategy') && ~isempty(self.opts.initialisation_strategy)
                strat = self.opts.initialisation_strategy;
            else
                strat = 'MAP';
            end            
            
            switch strat
                case 'random'
                    % Only Randomly Assign Unlabelled Data
                    labels = randi(self.nclust,self.Nx,1);
                    self.label(self.label == 0) = labels(self.label == 0);
                    
                    for nn = 1:self.nclust
                        
                        self.cluster{nn} = NIW(...
                            self.prior.m0,...
                            self.prior.k0,...
                            self.prior.n0,...
                            self.prior.S0,...
                            self.X(self.label==nn,:)...
                            );
                        
                    end
                    
                case 'MAP'
                    % Random order of data
                    inds = randperm(self.Nx);
                    
                    % Sample into clusters
                    for nn = 1:self.Nx
                        
                        ind = inds(nn);
                        if self.label(ind) == 0
                            % Get Data
                            x = self.X(ind,:);
                            
                            % Normalised posterior loglikelihoods
                            ll = self.nLL(self.predict_posterior(x));
                            % Greedy cluster
                            [~,ii] = max(exp(ll));
                            
                            % Add to cluster update labels
                            self.cluster{ii} = self.cluster{ii}.add_one(x);
                            self.label(inds(nn)) = ii;
                        end
                    end
                    
                case 'knn'
                    
                    % Random order of data
                    inds = find(self.label ~= 0);
                    inds_prime = find(self.label == 0);
                    
                    % Assign to Closest Neighbour
                    D2 = pdist2(self.X(inds_prime,:),self.X(inds,:));
                    [~,closest] = min(D2,[],2);
                    self.label(inds_prime) = self.label(inds(closest));
                    
                    for nn = 1:self.nclust
                        
                        self.cluster{nn} =NIW(...
                            self.prior.m0,...
                            self.prior.k0,...
                            self.prior.n0,...
                            self.prior.S0,...
                            self.X(self.label==nn,:)...
                            );
                        
                    end
                    
                    
                otherwise
                    error('Initialisation Strategy Not Implemented')
            end
            
            
        end
        
        function self = EM(self)

            % Max iterations in EM
            if isfield(self.opts,'em_steps') && ~isempty(self.opts.em_steps) && self.opts.em_steps > 0
                K = self.opts.em_steps;
            else
                K = 250;
            end
            
            % Initialise at labels 
            for cc = 1:self.nclust
                
                self.cluster{cc} = NIW(...
                    self.prior.m0,...
                    self.prior.k0,...
                    self.prior.n0,...
                    self.prior.S0,...
                    self.X(self.fixed==true & self.label==cc,:)...
                    );

            end

            for nn = 1:self.Nx
               
                if self.fixed(nn) == false
                   
                    ll = self.nLL(self.predict_posterior(self.X(nn,:)));
                    ii = self.nclust+1-sum(rand() < cumsum(exp(ll)));
                        
                    % Add to cluster update labels
                    self.cluster{ii} = self.cluster{ii}.add_one(self.X(nn,:));
                    self.label(nn) = ii;
                    
                end
                
            end
            
            loglik = NaN(self.Nx,self.nclust,K);
            for kk = 1:K
                
                % E Step
                
                % Responsibility                
                loglik(:,:,kk) = self.predict_posterior(self.X);
                r = exp(self.nLL(loglik(:,:,kk)')');
                
                
                if kk > 1 &&  LL_ALL < LL_OLD%sum(loglik(:,:,kk),'all') <= sum(loglik(:,:,kk-1),'all')
                    fprintf('EM converged after %i iterations\n',kk)
%                     break
                end                    
                
                % M Step
                for cc = 1:self.nclust
                    
                    self.cluster{cc} = NIW(...
                        self.prior.m0,...
                        self.prior.k0,...
                        self.prior.n0,...
                        self.prior.S0,...
                        self.X(self.fixed==true & self.label==cc,:)...
                        );
                    self.cluster{cc} = self.cluster{cc}.add_responsible_data(self.X(self.fixed == 0,:),r(self.fixed == 0,cc));
                    
                end
               
               
               
            end
            
            self.label(self.fixed==false) = self.predict_map(self.X(self.fixed==false,:));
            
            figure
            plot(squeeze(sum(loglik,[1,2])))

        end
        
        function self = gibbs_inference(self)
            
            % Collapsed Gibbs Sampling
            
            if isfield(self.opts,'gibbs_steps') && ~isempty(self.opts.gibbs_steps) && self.opts.gibbs_steps > 0
                K = self.opts.gibbs_steps;
            else
                K = 1000;
            end
            
            sampled_labels = NaN(self.Nx,K+1);
            sampled_labels(:,1) = self.label;
            sampled_labels(self.fixed,:) = repmat(sampled_labels(self.fixed,1),1,K+1);
            
            for kk = 1:K
                
                rndinds = 1:self.N;%randperm(self.N);
                
                for nn = 1:self.N
                    
                    if ~self.fixed(rndinds(nn))
                        x = self.X(rndinds(nn),:);
                        y = sampled_labels(rndinds(nn),kk);
                        
                        self.cluster{y} = self.cluster{y}.rem_one(x);
                        
                        % Normalised posterior loglikelihoods
                        ll = self.nLL(self.predict_posterior(x));
                        % Sample new cluster
                        ii = self.nclust+1-sum(rand() < cumsum(exp(ll)));
                        
                        % Add to cluster update labels
                        self.cluster{ii} = self.cluster{ii}.add_one(x);
                        sampled_labels(rndinds(nn),kk+1) = ii;
                    end
                    
                end
                
                self.label = sampled_labels(:,kk+1);
                
            end
            
            label_counts = NaN(self.Nx,self.nclust);
            for nn = 1:self.nclust
                label_counts(:,nn) = sum(sampled_labels(:,floor(K/2):end)==nn,2);
            end
            
            self.responsibility = label_counts./sum(label_counts,2);
            
            
            
            [~,self.label] = max(label_counts,[],2);
            for nn = 1:self.nclust
                
                self.cluster{nn} = NIW(...
                    self.prior.m0,...
                    self.prior.k0,...
                    self.prior.n0,...
                    self.prior.S0,...
                    self.X(self.label==nn,:)...
                    );
                
            end
            
            if self.debug
                figure
                gscatter(self.X(:,1),self.X(:,2),self.label)
                hold on
                for nn = 1:self.nclust
                    [mm,SS] = self.cluster{nn}.MAP();
                    plot_clusters(mm,SS)
                end
            end
            
        end
        
        
        function [mu,Sig] = MAP(self)
            
            mu = NaN(self.nclust,self.D);
            Sig = NaN(self.D,self.D,self.nclust);
            
            for nn = 1:self.nclust
                [m,S] = self.cluster{nn}.MAP();
                mu(nn,:) = m;
                Sig(:,:,nn) = S;
            end
            
        end
        
        function cluster = responsible_clusters(self)
            
            cluster = cell(1,self.nclust);
            for nn = 1:self.nclust
                cluster{nn} = NIW(...
                    self.prior.m0,...
                    self.prior.k0,...
                    self.prior.n0,...
                    self.prior.S0);
                cluster{nn} = cluster{nn}.add_responsible_data(...
                    self.X(self.responsibility(:,nn)~=0,:),...
                    self.responsibility(self.responsibility(:,nn)~=0,nn)...
                    );
            end
        end
        
        function [mu,Sig] = responsible_MAP(self)
            
            
            cluster = self.responsible_clusters();
            mu = NaN(self.nclust,self.D);
            Sig = NaN(self.D,self.D,self.nclust);
            
            for nn = 1:self.nclust
                [mu(nn,:),Sig(:,:,nn)] = cluster{nn}.MAP();
                %               % MANUAL BATCH UPDATE FOR DEBUG
                %                 rk = sum(self.responsibility(:,nn));
                %                 xbark = sum(self.X.*self.responsibility(:,nn))/rk;
                %                 mu(nn,:) = (rk*xbark + self.prior.k0*self.prior.m0)/(rk+self.prior.k0);
                %                 SkL = sqrt(self.responsibility(:,nn)).*(self.X-xbark);
                %                 Sk = SkL'*SkL;
                %                 Sig(:,:,nn) = (self.prior.S0 +...
                %                     Sk +...
                %                     (self.prior.k0*rk)/(self.prior.k0 + rk)*(xbark-self.prior.m0)'*(xbark-self.prior.m0))/...
                %                     (self.prior.n0 + rk + self.D + 2);
            end
            
        end
        
        function [ll] = predict_posterior(self,Xt)
            
            nP = size(Xt,1);
            ll = NaN(nP,self.nclust);
            
            for kk = 1:nP
                
                ll(kk,:) = cellfun(@(a) a.logpredpdf(Xt(kk,:)), self.cluster)...
                    + log(self.pi);
                
            end
            
        end
        
        function [yp] = predict_map(self,Xt)
            
            ll = self.predict_posterior(Xt);
            [~,yp] = max(ll,[],2);
            
        end
        
        
        function [ll] = predict_responsible_posterior(self,Xt)
            
            nP = size(Xt,1);
            ll = NaN(nP,self.nclust);
            cluster = self.responsible_clusters();
            
            for kk = 1:nP
                
                ll(kk,:) = cellfun(@(a) a.logpredpdf(Xt(kk,:)), cluster)...
                    + log(cellfun(@(a) a.N, cluster));
                
            end
            
        end
        
        function [yp] = predict_responsible_map(self,Xt)
            
            ll = self.predict_responsible_posterior(Xt);
            [~,yp] = max(ll,[],2);
            
        end
        
        
        
        % Dependent Getters
        
        function N = get.N(self)
            N = sum(cellfun(@(a) a.N,self.cluster));
        end
        
        function Nx = get.Nx(self)
            Nx = size(self.X,1);
        end
        
        
        function n = get.nclust(self)
            
            n = length(self.cluster);
            
        end
        
        function pi = get.pi(self)
            
            NsA = sum(cellfun(@(a) a.N,self.cluster)) + sum(self.prior.alpha);
            pi = (cellfun(@(a) a.N, self.cluster) + self.prior.alpha )/NsA;
            
        end
        
        %% Plotting
    
        function [] = plot_clusters(self)
            
            cmap = [0.8941    0.1020    0.1098
                    0.2157    0.4941    0.7216
                    0.3020    0.6863    0.2902
                    0.5961    0.3059    0.6392
                    1.0000    0.4980         0
                    1.0000    1.0000    0.2000
                    0.6510    0.3373    0.1569
                    0.9686    0.5059    0.7490
                    0.6000    0.6000    0.6000
                    0.8941    0.1020    0.1098];
                
            
            [mm,SS] = self.MAP();
            for kk = 1:self.nclust
                hold on
                if kk <=10
                    plot_clusters(mm(kk,:),SS(:,:,kk),cmap(mod(kk,11),:),'--')
                else
                    plot_clusters(mm(kk,:),SS(:,:,kk),'k','--')
                end                    
            end
            
            hold off
                      
        end
        
    end
    
    
    
    
    %% Static Methods
    methods(Static)
        
        function acc = accuracy(yp,yt)
            
            acc = sum(yp==yt)/size(yt,1);
            
        end
        
        function lse = logsumexp(A)
            
            lse = log(sum(exp(A-max(A)))) + max(A);
            
        end
        
        function ll = nLL(ll)
            
            ll = ll-GMM_unsupervised.logsumexp(ll);
            
        end
        
        
    end
    
    
    
end