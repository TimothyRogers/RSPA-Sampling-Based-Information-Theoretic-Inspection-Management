classdef NIW
    % Ignore property warning
    %#ok<*PROPLC>
    
    properties
        
        m0
        k0
        n0
        S0
        X = []
        
    end
    
    properties (SetAccess=private)
        
        N
        D
        mn
        kn
        nn
        Sn
        
    end
    
    methods
        
        function self = NIW(m0,k0,n0,S0,X)
            
            % A Normal Inverse Wishart Distribution
            % mu, Sig ~ NIW(m0,k0,n0,S0)
            % mu ~ N(m0,Sig/k0)
            % Sig ~ IW(S0,n0)
            
            self.m0 = m0;
            self.n0 = n0;
            self.S0 = S0;
            self.k0 = k0;
            
            self.mn = m0;
            self.nn = n0;
            self.Sn = S0 + k0*(m0'*m0);
            self.kn = k0;
            self.D  = size(m0,2);
            self.N  = 0;
            
            if nargin > 4
                
                self = self.add_data(X);
                
            end
            
        end
        
        function [mu,Sig] = MAP(self)
            
            mu = self.mn;
            Sig = self.Sn/(self.nn+self.D+2);
            
        end

        function self = add_data(self,X)
            
            nX = size(X,1);
            for nn = 1:nX
                self = self.add_one(X(nn,:));
            end
            
        end
        
        function self = add_one(self,x)
            
            self.N = self.N + 1;
            self.kn = self.kn + 1;
            self.mn = ((self.kn-1)*self.mn + x)/self.kn;
            
            self.nn = self.nn + 1;
            self.Sn = self.Sn + self.kn/(self.kn-1)*(x-self.mn)'*(x-self.mn);
            
        end
        
        function self = rem_data(self,X)
            
            nX = size(X,1);
            for nn = 1:nX
                self = self.rem_one(X(nn,:));
            end
            
        end
        
        function self = rem_one(self,x)
            
            self.N = self.N-1;
            self.Sn = self.Sn - self.kn/(self.kn-1)*(x-self.mn)'*(x-self.mn);

            
            self.kn = self.kn - 1;
            self.mn = ((self.kn+1)*self.mn-x)/self.kn;
            
            self.nn = self.nn - 1;
            
        end
        
        function self = add_responsible_data(self,X,r)
            % X is set of data N x D
            % r is responsibility vector N x 1
            
            nX = size(X,1);
            for nn = 1:nX
               
                self = self.add_responsible_one(X(nn,:),r(nn));
                
            end
            
        end
        
        function self = add_responsible_one(self,x,r)
            
            rx = r*x;
           
            self.N = self.N + r;
            self.kn = self.kn + r;
            self.nn = self.nn + r;
            
            self.mn = ((self.kn-r)*self.mn + rx)/self.kn;
            self.Sn = self.Sn + r*((self.kn)/(self.kn-r)*(x-self.mn)'*(x-self.mn));

        end
        
                
        function self = rem_responsible_data(self,X,r)
            % X is set of data N x D
            % r is responsibility vector N x 1
            
            nX = size(X,1);
            for nn = 1:nX
               
                self = self.rem_responsible_one(X(nn,:),r(nn));
                
            end
            
        end
        
        function self = rem_responsible_one(self,x,r)
            
            rx = r*x;
            
            self.Sn = self.Sn - r*self.kn/(self.kn-r)*(x-self.mn)'*(x-self.mn);

            self.N = self.N - r;
            self.kn = self.kn - r;
            self.nn = self.nn - r;
            self.mn = ((self.kn+r)*self.mn - rx)/self.kn;

        end
       
        
        
        function ll = logpredpdf(self,x)
            
            % Log predictive pdf is multivariate T
            % p( x* | X ) = T( m_n, (k_n+1)/(k_n*nu_prime)*S_n, nu_prime)
            % nu_n' = nu_n - D + 1
            % Murphy PML 2021 Eq 7.144 pp. 199
            
            D = self.D;
            nu_prime = self.nn - self.D + 1;
            Sig = self.Sn*(self.kn+1)/(self.kn*nu_prime);
            U = chol(Sig);
            res = x-self.mn;
            QU = U'\res'; % sqrtm(Sig^-1)*res';
            
            ll = gammaln((nu_prime+D)/2) -...
                gammaln(nu_prime/2) -...
                D/2*log(nu_prime) -...
                D/2*log(pi) -...
                sum(log(diag(U))) - ... % 0.5 * log(det(self.Sn))
                (nu_prime+D)/2 * log(1 + (QU'*QU)/nu_prime );
            
        end
        
        
    end
end