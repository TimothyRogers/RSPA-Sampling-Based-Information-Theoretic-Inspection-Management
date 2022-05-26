classdef DPGMM_semi_supervised
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
        user_keys
        fixed
        label_map
        
    end
    
    properties (Dependent)
        nclust
        Pi % Mixing proportions
        N
        Nx
        keys
        nkeys
        unique_keys
        unique_labels
    end
    
    methods
        
        function self = DPGMM_semi_supervised(X, labels, prior, opts)
            
            if nargin > 3
                self.opts = opts;
            end
            
            if nargin > 2
                self.prior = prior;
            else
                error('Must Supply Prior')
            end
            
            nclust = 0;
            self.cluster = {};
            self.D = size(X,2);
            self.X = X;
            
            if nargin > 1
                self.user_keys = labels;
            else
                self.user_keys = NaN(self.N,1);
            end
            
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
            
            % Initialise DPGMM based on available labels
            
            % Sorted list of user provided keys
            uni_keys = sort(unique(self.user_keys));
            uni_keys = uni_keys(~isnan(uni_keys));
            % How many keys
            nkeys = length(uni_keys);
            % Build label map
            self.label_map = [uni_keys,(1:nkeys)'];
            self.fixed = zeros(self.Nx,1);
            
            % Create nkeys clusters
            for kk = 1:nkeys
                key_locs = self.findKey(uni_keys(kk));
                self.cluster{self.getLabel(uni_keys(kk))} = NIW(...
                    self.prior.m0,...
                    self.prior.k0,...
                    self.prior.n0,...
                    self.prior.S0,...
                    self.X(key_locs,:)...
                    );
                self.label(key_locs) = self.getLabel(uni_keys(kk));
                self.fixed(key_locs) = 1;
            end
            
            
            if self.opts.shuf
                inds = randperm(self.Nx); 
            else
                inds = 1:self.Nx;
            end
            
            
            % Assign other points by sampling
            for nn = 1:self.Nx
                
                pnt = inds(nn);
                
                % If Data point does not have a label skip this datum
                if self.fixed(pnt) == true
                    continue
                end
                
                
                % Normalised posterior loglikelihoods
                ll = self.nLL(self.predict_posterior(self.X(pnt,:)));
                % Sample new cluster
                ii = self.nclust+2-sum(rand() < cumsum(exp(ll)));
                
                % Assign to cluster
                if ii > self.nclust
                    self.cluster{ii} = self.prior_clust.add_one(self.X(pnt,:));
                else
                    self.cluster{ii} = self.cluster{ii}.add_one(self.X(pnt,:));
                end
                self.label(pnt) = ii;
                
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
                
                rndinds = 1:self.Nx;%randperm(self.N);
                
                for nn = 1:self.Nx
                                        
                    if self.fixed(rndinds(nn)) == false
                        
                        
                        % Get data and label for this current point
                        x = self.X(rndinds(nn),:);
                        y = self.label(rndinds(nn));
                        
                        % If cluster has only one datapoint associated then
                        % remove cluster else remove data
                        if self.cluster{y}.N == 1
                            self.cluster = self.cluster(setdiff(1:self.nclust,y));
                            self.label(self.label>y) = self.label(self.label>y) - 1;
                            self.dropLabel(y);
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
                    
                end
                
                nlabels(kk) = self.nclust;
                
            end
            
            sampled_labels(self.fixed == true,:) = ...
                repmat(self.label(self.fixed == true)',1,K+1);
            
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
                    + log(self.Pi);
                
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
        
        %% Label Map Management Functions
        
        % The label map is one of the most confusing things in this code
        % base (or ever). We wish to manage the mismatch between labels
        % that may be supplied by a practitioner and the indices of the
        % clusters inside the DP. The main challenge comes from the fact
        % that the internal indices of the clusters can and will change as
        % the DP learns the data distribution. A number of functions here
        % manage this mismatch that we see by maintaining a simple mapping
        % between the user assigned labels and the internal indices.
        %
        % A little terminology which will be useful to have for reference
        % later.
        %
        % key - the user supplied quantity
        % label - the internal cluster index
        %
        % self.user_key  - the vector of user supplied labels
        % self.labels    - the list of internal indices each datum is
        %                  assosciated with
        % self.label_map - a L x 2 array, column 1 contains user keys and
        %                  column 2 a corresponding internal label
        % self.fixed     - vector boolean indicating if user has labelled
        %                  this point
        %
        %
        
        
        function [inMap] = isKey(self,key)
            % Is this user key currently in the map
            inMap = any(self.label_map(:,1) == key);
        end
        
        function [inMap] = isLabel(self,label)
            % Is this user key currently in the map
            inMap = any(self.label_map(:,2) == label);
        end
        
        function [key] = getKey(self,label)
            % Get key related to corresponding label
            key = self.label_map(self.label_map(:,2)==label,1);
        end
        
        function [label] = getLabel(self,key)
            % Get label related to corresponding key
            label = self.label_map(self.label_map(:,1)==key,2);
        end
        
        
        function [keyInds] = findKey(self,key)
            % Return indices of this key
            keyInds = self.user_keys(:,1) == key;
        end
        
        function [labelInds] = findLabel(self,label)
            % Return indices of this key
            labelInds = self.label(:,2) == label;
        end
        
        function [self] = updateKey(self,key_old,key_new)
            % Replace the old key with the new one
            self.label_map(findKey(key_old),1) = key_new;
        end
        
        function [self] = updateLabel(self,label_old,label_new)
            % Replace the old label with the new one
            self.label_map(findLabel(label_old),2) = label_new;
        end
        
        function [self] = dropLabel(self,label)
            % Remove label from the map and decrement all labels that are
            % one higher to maintain a contiguous sequence
            %
            % This will be useful when removing a cluster from the DP
            if any(self.label_map(:,2) == label)
                error('Should not have an empty cluster for which there are user labels')
            end
            
            self.label_map(self.label_map(:,2) > label,2) = ...
                self.label_map(self.label_map(:,2) > label,2) - 1;
        end
        
        function [keys] = getKeys(self,labels)
            
            Np = size(labels,1);
            keys = NaN(Np,1);
            for nn = 1:Np
                if self.isLabel(labels(nn))
                    keys(nn) = self.getKey(labels(nn));
                else
                    keys(nn) = -labels(nn);
                end
            end
            
        end
        
        %% Dependent Getters
        
        function N = get.N(self)
            % Number of data points in clusters
            N = sum(cellfun(@(a) a.N,self.cluster));
        end
        
        function Nx = get.Nx(self)
            % Total number of data loaded into the model
            Nx = size(self.X,1);
        end
        
        
        function n = get.nclust(self)
            % Number of clusters
            n = length(self.cluster);
            
        end
        
        function Pi = get.Pi(self)
            % Mixing proportions
            NsA = sum(cellfun(@(a) a.N,self.cluster)) + sum(self.prior.alpha);
            Pi = [(cellfun(@(a) a.N, self.cluster)),self.prior.alpha]./NsA;
            
        end
        
        function nkeys = get.nkeys(self)
            % Number of keys = size of map
            nkeys = size(self.label_map,1);
        end
        
        function unique_keys = get.unique_keys(self)
            % Unique keys is just all the keys in the map
            unique_keys = self.label_map(:,1);
        end
        
        function unique_labels = get.unique_labels(self)
            % Unique labels is just all the labels in the map
            % N.B. this should be integers 1:nkeys
            unique_labels = self.label_map(:,2);
        end
        
        function keys = get.keys(self)
            % Return labels as user keys, NaN if in cluster with no key
            keys = NaN(self.Nx,1);
            for kk = 1:self.nkeys
                keys(self.label == self.getLabel(self.unique_keys(kk))) = self.unique_keys(kk);
            end
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