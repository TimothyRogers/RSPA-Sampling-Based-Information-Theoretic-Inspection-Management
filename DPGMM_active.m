classdef DPGMM_active
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
        debug = true;
        sampled_labels
        user_keys
        fixed
        label_map
        K_samp
        query_prob
        
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
        
        function self = DPGMM_active(X, labels, prior, opts)
            
            % The active version is a bit of a different beast. Labels are
            % supplied for all datapoints but these are not used until
            % queried. This is research code so obviously this wouldn't
            % work in a practial application
            
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
            self.fixed = zeros(self.Nx,1);
            self.label = zeros(self.Nx,1);
            
            if nargin > 1
                self.user_keys = labels;
            else
                error('Labels must be provided for active DP')
            end
            
            self.prior_clust = NIW(...
                self.prior.m0,...
                self.prior.k0,...
                self.prior.n0,...
                self.prior.S0...
                );
            
            self = self.validate_opts();
            
        end
        
        function self = validate_opts(self)
            
            % Number of Gibbs sampler steps
            if ~isfield(self.opts,'gibbs_steps')
                self.opts.gibbs_steps = 1000;
            elseif isempty(self.opts.gibbs_steps) || self.opts.gibbs_steps <= 0
                warning('Invalid number of Gibbs steps, setting to default K=1000')
                self.opts.gibbs_steps = 1000;
            end
            
            % Size of window for backwards sampling
            if ~isfield(self.opts,'backsamp_window')
                self.opts.backsamp_window = -1; % -1 is whole length
            elseif isempty(self.opts.backsamp_window) ||...
                    self.opts.backsamp_window < -1 ||...
                    self.opts.backsamp_window > self.Nx
                warning('Invalid backsampling window length, set to default -1')
                self.opts.backsamp_window = -1; % -1 is whole length
            end
            
            % Number of loops in backsampling
            if ~isfield(self.opts,'backsamp_loops')
                self.opts.backsamp_loops = 5; 
            elseif isempty(self.opts.backsamp_loops) ||...
                    self.opts.backsamp_loops < 0
                warning('Invalid backsampling window length, set to default -1')
                self.opts.backsamp_loops = 5; 
            else
                self.opts.backsamp_loops = floor(self.opts.backsamp_loops);
            end
            
            
            
        end
        
        function self = add_data(self,X)
            %
            error('Not implemented')
            
            % Need to manage additional labels etc.
            self.X = [self.X;X];
            
        end
        
        function self = rem_data(self,X,y) %#ok<INUSD>
            error('Who wants to remove data from their model anyway? Weirdo.')
        end
        
        
        function self = add_point(self,pnt)
            
            label = self.label(pnt);
            
            % Assign to cluster
            if self.label(pnt) ==  self.nclust+1
                self.cluster{label} = self.prior_clust.add_one(self.X(pnt,:));
            elseif self.label(pnt) > self.nclust+1
                error('Putting into cluster K+2!')
            else
                self.cluster{label} = self.cluster{label}.add_one(self.X(pnt,:));
            end
            
        end
        
        function self = rem_point(self,pnt)
            
            label = self.label(pnt);
            
            % If cluster has only one datapoint associated then
            % remove cluster else remove data
            if self.cluster{label}.N == 1
                self.cluster = self.cluster(setdiff(1:self.nclust,label));
                self.label(self.label>label) = self.label(self.label>label) - 1;
                self = self.dropLabel(label);
            else
                self.cluster{label} = self.cluster{label}.rem_one(self.X(pnt,:));
            end
        end
        
        
        function self = initialise(self)
            
            % Initialise Active DPGMM
            
            % We will sequentially add the data applying the online Gibbs
            % approach
            
            if self.opts.shuf == true
                inds = randperm(self.Nx); % Random indices
            else
                inds = 1:self.Nx; % Original order
            end
            self.K_samp = NaN(self.Nx,1);
            self.query_prob = NaN(self.Nx,1);
            
            % Outer loop is adding data
            for nn = 1:self.Nx
                
                % This data point in nn in inds
                pnt = inds(nn);
                
                % If Data point does not have a label skip this datum
                if self.fixed(pnt) == true
                    error('Unseen datapoint already fixed, iteration: %i, datapoint: %i\n',nn,pnt)
                end
                
                
                % Normalised posterior loglikelihoods
                ll = self.nLL(self.predict_posterior(self.X(pnt,:)));
                % Sample new cluster
                ii = self.nclust+2-sum(rand() < cumsum(exp(ll)));
                
                % Should we query this datapoint?
                self.query_prob(nn) = self.calc_query_prob(ll); % STORE QP WITH DATUM???
   
                if rand < self.query_prob(nn) % Point queried
                    fprintf('Query requested on datapoint %i in iteration %i\n',pnt,nn)
                    self.fixed(pnt) = true; % This point has been observed
                    self = self.add_queried_point(pnt,ii);
                else % Point not queried
                    self.label(pnt) = ii;
                    self = self.add_point(pnt);
                end
                
                % Inner loop is backsampling, i.e. Gibbs learn from
                % previous posterior as the prior
                for tmp = 1:self.opts.backsamp_loops
                    % Randomise datapoints up to this one for backsampling
                    if self.opts.backsamp_window == -1
                        ss = 1;
                    else
                        ss = max(nn-self.opts.backsamp_window+1,1);
                    end
                    cont_inds = ss:nn;
                    inds_int = inds(cont_inds(randperm(nn-ss+1)));
                    for jj = 1:length(cont_inds)
                        
                        pnt = inds_int(jj);
                        
                        % Datapoint already labelled, do not resample.
                        if ~self.fixed(pnt)
                            
                            % Remove from cluster
                            self = self.rem_point(pnt);
                            
                            % Normalised posterior loglikelihoods
                            ll = self.nLL(self.predict_posterior(self.X(pnt,:)));
                            % Sample new cluster
                            ii = self.nclust+2-sum(rand() < cumsum(exp(ll)));
                            
                            % Add to cluster
                            self.label(pnt) = ii;
                            self = self.add_point(pnt);
                            
                        end
                        
                    end
                end
                self.K_samp(nn) = self.nclust;
                if mod(nn,100)==0; fprintf('Active DP Iteration: %i\n',nn), end
            end
            
        end
        
        function self = gibbs_inference(self)
            
            error('Not Yet Implemented')
            
          
            
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
            if isempty(self.label_map)
                inMap = false;
            else
                inMap = any(self.label_map(:,1) == key);
            end
        end
        
        function [inMap] = isLabel(self,label)
            % Is this user key currently in the map
            if isempty(self.label_map)
                inMap = false;
            else
                inMap = any(self.label_map(:,2) == label);
            end
        end
        
        function [key] = getKey(self,label)
            % Get key related to corresponding label
            if isempty(self.label_map)
                key = [];
            else
                key = self.label_map(self.label_map(:,2)==label,1);
            end
        end
        
        function [label] = getLabel(self,key)
            % Get label related to corresponding key
            if isempty(self.label_map)
                label = [];
            else
                label = self.label_map(self.label_map(:,1)==key,2);
            end
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
        
        function [self] = addKey(self,key,label)
            % Add a new key to the map
            self.label_map = [self.label_map; key, label];
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
        
        %% Querying Functions
        
        function [query_prob] = calc_query_prob(self,ll)
            
            % Calculate inspection probability as shannon efficiency
            nLL = self.nLL(ll); % Normalise loglikelihood
            shannon_entropies = -sum(exp(nLL).*nLL,2); % Calc Shannon entrop
            if self.nclust == 0 % First datapoint always query
                efficiency = 1;
            else
                efficiency = shannon_entropies./log(self.nclust+1); % Normalise for efficiency
            end
            
            query_prob = efficiency; % Return
            
%             oneVall = [max(nLL), log(1-exp(max(nLL)))];
%             efficiency = -sum(exp(oneVall).*oneVall,2)./log(2);
% 
%             query_prob = max(efficiency,exp(nLL(end)));
            
        end
        
        function [self] = add_queried_point(self,pnt,sampled_label)
            
            % The cluster associated with the user provided key if user key
            % for this datapoint is not in map assigned_label -> empty
            assigned_key = self.user_keys(pnt);
            assigned_label = self.getLabel(assigned_key);
            
            % User key associated with currently sampled label, if not in
            % map sampled_key -> empty
            sampled_key = self.getKey(sampled_label);
            
            % The current key is not in the map, i.e. we have not seen a
            % datpoint with this user key before.
            if isempty(assigned_label)
                
                % Is the cluster this datapoint has been put in already
                % labelled
                if self.isLabel(sampled_label)
                    
                    % The sampled label is already in the map but not
                    % associated with this key, i.e. a the cluster this
                    % point was put in already has a user label assigned to
                    % it.
                    %
                    % Pass to label clash routine.
                    
%                     sampled_key = self.getKey(sampled_label);
                    
                    [self,use_label] = self.label_clash(pnt,...
                        assigned_label,...
                        assigned_key,...
                        sampled_label,...
                        sampled_key);
                    
                else
                    % In this case the cluster label has no associated key
                    % so propagate label to all points in that cluster by
                    % adding to map
                    
                    self = self.addKey(assigned_key,sampled_label);
                    use_label = sampled_label;
                    self.label(pnt) = use_label;
                    self = self.add_point(pnt);
                    
                end
                
                
                
            else % We have seen data with this user key before
                
                if assigned_label == sampled_label
                    % The sampled label puts it in the cluster with all the
                    % other points having the same user key.
                    use_label = sampled_label;
                    self.label(pnt) = use_label;
                    self = self.add_point(pnt);
                else
                    % The sampled cluster is different from the one that
                    % datapoints currently associated with this user key
                    % are in.
                    %
                    % Pass to label clash routine.
                    
                    [self, use_label] = self.label_clash(pnt,...
                        assigned_label,...
                        assigned_key,...
                        sampled_label,...
                        sampled_key);
                    
                end
                
            end
            
            
            
            
            % Update the label of this point to associate it with the right
            % cluster to put the label in.
            %self.label(pnt) = use_label;
            %self = self.add_point(pnt);
            
            
            
        end
        
        function [self,use_label] = label_clash(self, pnt, assigned_label, assigned_key, sampled_label, sampled_key)
            
            fprintf('Label Clash on Datapoint %i\n',pnt)
            if isempty(assigned_label)
                use_label = self.nclust+1;
                self = self.addKey(assigned_key,use_label);
            else 
                % Brute force use assigned label
                use_label = assigned_label;
            end
            self.label(pnt) = use_label;
            self = self.add_point(pnt);
            
            
%             % Unlabelled points MAP update - fully supervised 2 component
%             % GMM with the clashing clusters
%             pnts = find((self.label == use_label | self.label == sampled_label) & self.fixed == 0);
%             
%             inds = randperm(length(pnts));
%             pnts = pnts(inds);
%             
%             % Remove all the unsupervised points from their clusters
%             for nn = 1:length(pnts)
%                 self = self.rem_point(pnts(nn));
%             end
%             
%             % Add in with MAP
%             for nn = 1:length(pnts)
%                 ll = self.predict_posterior(self.X(pnts(nn),:));
%                 if ll(sampled_label) > ll(use_label)
%                     ii = sampled_label;
%                 else
%                     ii = use_label;
%                 end                 
%                     
%                 self.label(pnts(nn)) = ii;
%                 self = self.add_point(pnts(nn));
%             end
            
            
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
                    plot_clusters(mm(kk,1:2),SS(1:2,1:2,kk),cmap(mod(kk,11),:),'--')
                else
                    plot_clusters(mm(kk,1:2),SS(1:2,1:2,kk),'k','--')
                end                    
            end
            
            hold off
                      
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
            
            ll = ll-DPGMM_active.logsumexp(ll);
            
        end
        
        
    end
    
    
    
end