classdef GMM_supervised
   
    properties
        
        prior
        cluster
        cluster_labels
        D
             
    end
    
    properties (Dependent)
        nclust
        pi % Mixing proportions
        N
    end
    
    methods 
        
        function self = GMM_supervised(X, y, prior)
            
           
            self.cluster_labels = unique(y);
            nclust = length(self.cluster_labels);
            self.cluster = {};
            self.D = size(X,2);
            
            if nargin > 2
                self.prior = prior;
            else
                error('Must Supply Prior')
            end
            
            for nn = 1:nclust
                
                self.cluster{nn} = NIW(...
                    self.prior.m0,...
                    self.prior.k0,...
                    self.prior.n0,...
                    self.prior.S0,...
                    X(y==self.cluster_labels(nn),:)...             
                    );
                
            end
            
        end
        
        function self = add_data(self,X,y)
       
            for nn = 1:self.nclust
                self.cluster{nn} = self.cluster{nn}.add_data(X(y==self.cluster_labels(nn),:));                
            end
            
        end
        
        function self = rem_data(self,X,y) %#ok<INUSD>
            error('Who wants to remove data from their model anyway? Weirdo.')
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
        
        
        
        % Dependent Getters
        
        function N = get.N(self)
            N = sum(cellfun(@(a) a.N,self.cluster));
        end
        
        function n = get.nclust(self)
           
            n = length(self.cluster);
            
        end
        
        function pi = get.pi(self)
            
            NsA = self.N + sum(self.prior.alpha);
            pi = (cellfun(@(a) a.N, self.cluster) + self.prior.alpha )/NsA;
            
        end
           
    end
    
    
    methods(Static)
     
        function acc = accuracy(yp,yt)
           
            acc = sum(yp==yt)/size(yt,1);
            
        end
        
        
    end
    
    
    
end