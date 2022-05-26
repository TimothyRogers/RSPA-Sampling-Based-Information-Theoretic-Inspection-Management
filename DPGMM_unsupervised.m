classdef DPGMM_unsupervised
    % Ignore property warning
    %#ok<*PROPLC>
    %#ok<*PROP>
    
    properties
        
        opts = struct();
        prior
        cluster
        prior_clust
        D
        X
        label
        responsibility
        debug = false;
        sampled_labels
        
    end
    
    properties (Dependent)
        nclust
        pi % Mixing proportions
        N
        Nx
    end
    
    methods
        
        function self = DPGMM_unsupervised(X, prior, opts)
            
            if nargin > 2
                self.opts = opts;
            end
            
            if nargin > 1
                self.prior = prior;
            else
                error('Must Supply Prior')
            end
            
            
            nclust = 0;
            self.cluster = {};
            self.D = size(X,2);
            self.X = X;
            
            self.prior_clust = NIW(...
                self.prior.m0,...
                self.prior.k0,...
                self.prior.n0,...
                self.prior.S0...
                );
        
            
        end
        
        function self = add_data(self,X)
            
            self.X = [self.X;X];
            
        end
        
        function self = rem_data(self,X,y) %#ok<INUSD>
            error('Who wants to remove data from their model anyway? Weirdo.')
        end
        
        function self = initialise(self)
            
            % Initialise DPGMM sequentially adding data
            
            % First point in first cluster
            self.cluster{1} = NIW(...
                self.prior.m0,...
                self.prior.k0,...
                self.prior.n0,...
                self.prior.S0,...
                self.X(1,:)...
            );
            self.label(1) = 1;
        
            
            % Assign other points by sampling
            for nn = 2:self.Nx
                
                % Normalised posterior loglikelihoods
                ll = self.nLL(self.predict_posterior(self.X(nn,:)));
                % Sample new cluster
                ii = self.nclust+2-sum(rand() < cumsum(exp(ll)));
                
                % Assign to cluster]
                if ii > self.nclust
                    self.cluster{ii} = self.prior_clust.add_one(self.X(nn,:));
                else
                    self.cluster{ii} = self.cluster{ii}.add_one(self.X(nn,:));
                end
                self.label(nn) = ii;
                
            end
            
            
            
        end
        
        function self = gibbs_inference(self)
            
            % Collapsed Gibbs Sampling for DPGMM
            
            if isfield(self.opts,'gibbs_steps') && ~isempty(self.opts.gibbs_steps) && self.opts.gibbs_steps > 0
                K = self.opts.gibbs_steps;
            else
                K = 1000;
            end
            
            sampled_labels = NaN(self.Nx,K+1);
            sampled_labels(:,1) = self.label;
            
            for kk = 1:K
                
                rndinds = randperm(self.N);
                
                for nn = 1:self.N
                    
                    x = self.X(rndinds(nn),:);
                    y = self.label(rndinds(nn));
                    
                    if self.cluster{y}.N == 1
                        self.cluster = self.cluster(setdiff(1:self.nclust,y));
                        self.label(self.label>y) = self.label(self.label>y) - 1;
                    else
                        self.cluster{y} = self.cluster{y}.rem_one(x);
                    end
                    
                    
                    % Normalised posterior loglikelihoods
                    ll = self.nLL(self.predict_posterior(x));
                    % Sample new cluster
                    ii = self.nclust+2-sum(rand() < cumsum(exp(ll)));
                    
                    % Check If New Cluster
                    if ii == self.nclust+1
                        self.cluster{ii} = self.prior_clust.add_one(x);
                    else
                        % Add to cluster update labels
                        self.cluster{ii} = self.cluster{ii}.add_one(x);
                    end

                    self.label(rndinds(nn)) = ii;
                    sampled_labels(rndinds(nn),kk+1) = ii;

                end
                
                nlabels(kk) = self.nclust;
                
            end
            
            self.sampled_labels = sampled_labels;
            
            max_clusts = max(nlabels,[],'all');
            label_counts = NaN(self.N,max_clusts);
            for nn = 1:max_clusts
                label_counts(:,nn) = sum(sampled_labels(:,round(K/2):end)==nn,2);
            end
            
            self.responsibility = label_counts./sum(label_counts,2);
            
            [~,self.label] = max(label_counts,[],2);
            nclust = length(unique(self.label));
            for nn = 1:nclust
                
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
                histogram(nlabels(round(K/2):end))
                xlabel('Cluster Label Unique Components')
                
                figure
                gscatter(self.X(:,1),self.X(:,2),self.label)
                hold on
                for nn = 1:self.nclust
                    [mm,SS] = self.cluster{nn}.MAP();
                    plot_clusters(mm,SS)
                end
            end
            
        end
        
        %% Cluster Estimates for Plotting
        
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
            
            max_clusts = size(self.responsibility,2);
            cluster = cell(1,max_clusts); 
            
            for nn = 1:max_clusts
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
            mu = NaN(size(cluster,2),self.D);
            Sig = NaN(self.D,self.D,size(cluster,2));
            
            for nn = 1:size(cluster,2)
                [mu(nn,:),Sig(:,:,nn)] = cluster{nn}.MAP();
            end
            
        end
        
        %% Prediction Functions
        
        function [ll] = predict_posterior(self,Xt)
            
            nP = size(Xt,1);
            ll = NaN(nP,self.nclust+1);
            
            for kk = 1:nP
                
                ll(kk,:) = [cellfun(@(a) a.logpredpdf(Xt(kk,:)), self.cluster),...
                    self.prior_clust.logpredpdf(Xt(kk,:))]...
                    + log(self.pi);
                
            end
            
        end
        
        function [yp] = predict_map(self,Xt)
            
            ll = self.predict_posterior(Xt);
            [~,yp] = max(ll,[],2);
            
        end
        
        
        function [ll] = predict_responsible_posterior(self,Xt)
            
            nP = size(Xt,1);
           
            cluster = self.responsible_clusters();
            ll = NaN(nP,size(cluster,2));
            
            for kk = 1:nP
                
                ll(kk,:) = cellfun(@(a) a.logpredpdf(Xt(kk,:)), cluster)...
                    + log(cellfun(@(a) a.N, cluster));
                
            end
            
        end
        
        function [yp] = predict_responsible_map(self,Xt)
            
            ll = self.predict_responsible_posterior(Xt);
            [~,yp] = max(ll,[],2);
            
        end
        
        
        
        %% Dependent Getters
        
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
            pi = [(cellfun(@(a) a.N, self.cluster)),self.prior.alpha]./NsA;
            
        end
        
    end
    
    
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