%% Active DP Experiments
% 
% 
% We will compare the active DP to a fully supervised and semi-supervised GMM 
% with random label assignments.

clear 
close all
clc
%%

% % 17 class lanl
% load('4dof_feats2')
% inputs = [];
% for ch = 4
%     inputs = [inputs,CH{ch}(:,1:2),CT{ch}(:,1:2)]; %#ok<AGROW>
% end
% labs = labels(:,1);
% f1 = 1; % features for visualising
% f2 = 4; 
% 
% x = inputs;
% y = labs;
% x_test = inputs;
% y_test = labs;

% load('los_alamos_4dof_features.mat')
% 
% 
% energy = [features.sums];
% 
% D_full = [];
% 
% dims_frf = 1:2;
% dims_coh = 1:2;
% % D_full = [proj_frf{1}(dims,:)',proj_frf{2}(dims,:)',proj_frf{3}(dims,:)',proj_frf{4}(dims,:)'];
% for sensor = [2,4]
%     D_full = [D_full S_frf{sensor}(:,dims_frf)];
% %     D_full = [D_full proj_frf{sensor}(dims_frf,:)'];
% %     D_full = [D_full energy(sensor,:)'];
%     D_full = [D_full S_coh{sensor}(:,dims_coh)];
% %     D_full = [D_full proj_coh{sensor}(dims_coh,:)'];
%     
% end
% D_full = [D_full, energy(:,:)'];
% 
% % clear D_full
% for sensor = 1:4
%     for state = 1:17
%         pmags((1+(state-1)*50):state*50,(1+(sensor-1)*3):sensor*3) = squeeze(features(state).pmag(:,sensor,:))';
%         wnhzs((1+(state-1)*50):state*50,(1+(sensor-1)*3):sensor*3) = squeeze(features(state).wnhz(:,sensor,:))';
%         rmss((1+(state-1)*50):state*50,(1+(sensor-1)):sensor) = squeeze(features(state).rms(sensor,:));
%     end
% end

load('D_z24')
D_full = D_z24(:,1:4);


mD = nanmean(D_full);
sD = nanstd(D_full);
D_full = bsxfun(@rdivide,bsxfun(@minus,D_full,mD),sD);

D = D_full';
x = D_full;
x_test = D_full;
% y = reshape(repmat(1:17,50,1),850,1);
% y_test = ;
y = D_z24(:,end);
y_test = D_z24(:,end);

% Colormaps
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
cmap = cmap(setdiff(1:10,6),:);

cmap = jet(17);
%% 
% Data is the usual Lawrence drifting MDOF dataset with 6 classes. 2010 datapoints 
% in the test set and 990 in the training set.
% 
% Let's visualise the data

figure
for kk = 1:length(unique(y))
    scatter(x(y==kk,1),x(y==kk,2),2,cmap(kk,:),'filled')
    scatter(x_test(y_test==kk,1),x_test(y_test==kk,2),2,cmap(kk,:),'Marker','d')
    hold on
    xlabel('X1')
    xlabel('X2')
end
%% 
% Figrue above shows all the training data as filled dots and test data as diamonds.
%% Specification of the prior
% 
% 
% Using an emiprical Bayes prior based on the data, i.e. all one big Gaussian. 
% We will use this prior in all the experiments

% x_test = (x_test - mean(x))./std(x);
% x = (x-mean(x))./std(x);


prior.m0 = mean(x);
prior.k0 = 1;
prior.n0 = size(x,2);

% D = x(1:50,:)';
% [d,n] = size(D);
% mu = nanmean(D,2);
% Do = bsxfun(@minus,D,mu);
% ss = sum(Do(:).^2)/(d*n);
% prior.S0 = (prior.n0+size(prior.m0,2))*cov(x);
prior.S0 = (prior.n0+size(prior.m0,2))*eye(size(x,2));
% prior.S0 = eye(size(x,2));
% prior.S0 = (prior.n0+size(prior.m0,2))*cov(x(1:50,:));
% prior.S0 = ss*eye(d);
%%
prior.alpha = ones(1,length(unique(y)))*length(y)/length(unique(y));

%% 
% We can specify and plot the prior, it is shown here with the training data.

C = NIW(prior.m0,prior.k0,prior.n0,prior.S0);
[mu_prior,Sig_prior] = C.MAP();

figure
for kk = 1:length(unique(y))
    scatter(x(y==kk,1),x(y==kk,2),2,cmap(kk,:),'filled')
    hold on
    xlabel('X1')
    xlabel('X2')
end
plot_clusters(mu_prior(:,1:2),Sig_prior(1:2,1:2,:))
%% 
% 
%% Fully Supervised GMM
% 
% 
% In the fully supervised case, all the data are available for labelling then 
% this can be compared with the test set.

%% Fit supervised model
supervised = GMM_supervised(x,y,prior);
% Pull out the MAP posterior cluster
[mu_post,Sig_post] = supervised.MAP();
% Plot as Before
figure
for kk = 1:length(unique(y))
    scatter(x(y==kk,1),x(y==kk,2),2,cmap(kk,:),'filled')
    hold on
    xlabel('X1')
    xlabel('X2')
    plot_clusters(mu_post(kk,1:2),Sig_post(1:2,1:2,kk),cmap(kk,:))
end
plot_clusters(mu_prior(:,1:2),Sig_prior(1:2,1:2,:))
%% 
% We can then pull out the test accuracy by looking at the MAP label assignment 
% of the new data |x_test|

yp = supervised.predict_map(x_test);

acc_supervised = supervised.accuracy(yp,y_test);
fprintf('Test Accuracy: %.1f%%\n',acc_supervised*100)
%% 
% We can see good test performance from the fully supervised model, maybe unserprisingly 
% but this gives a good benchmark for the rest of our experiements.
% 
% 
%% Unsupervised GMM

opts.gibbs_steps = 250;

unsupervised = DPGMM_unsupervised(x,prior,opts);
unsupervised = unsupervised.initialise();
unsupervised = unsupervised.gibbs_inference();
% figure
% unsupervised.plot_clusters();

yp = unsupervised.predict_map(x_test);
acc_unsupervised = unsupervised.accuracy(yp,y_test);
fprintf('Test Accuracy: %.1f%%\n',acc_unsupervised*100)
%% 
% 
%% Supervised GMM - Random Subsets

nrepeats = 0;
nstep = 100;

subset = cell(nstep,nrepeats);
breaks = floor(linspace(20,length(y),nstep));
acc_subset = NaN(nstep,nrepeats);   


for rr = 1:nrepeats
    parfor pp = 2:nstep
        
        inds = randperm(length(y));
        
        
        
        labels = zeros(size(y));
        labels(inds(1:breaks(pp))) = y(inds(1:breaks(pp)));
        
        subset{pp,rr} = GMM_semi_supervised(x,labels,prior);
        
        yp = subset{pp,rr}.predict_map(x_test);
        
        acc_subset(pp,rr) = subset{pp,rr}.accuracy(yp,y_test);
        
    end
end
%% 
% 
%% Semi-Supervised GMM - Random Sampling
% 
% 
% One important comparison that we can make is the performance of the fully 
% supervised GMM with a semi-supervised model where a proportion of the data are 
% selected at random to be inspected.


% semi_supervised = cell(nstep,nrepeats);
% breaks = floor(linspace(0,length(y),nstep));
% acc_semi_supervised = NaN(nstep,nrepeats);   
% opts.gibbs_steps = 250;
% 
% for rr = 1:nrepeats
%     parfor pp = 1:nstep
%         
%         inds = randperm(length(y));
%         
%         labels = zeros(size(y));
%         labels(inds(1:breaks(pp))) = y(inds(1:breaks(pp)));
%         
%         semi_supervised{pp,rr} = GMM_semi_supervised(x,labels,prior,opts);
%         semi_supervised{pp,rr} = semi_supervised{pp,rr}.initialise();
%         semi_supervised{pp,rr} = semi_supervised{pp,rr}.gibbs_inference();
%         
%         yp = semi_supervised{pp,rr}.predict_map(x_test);
%         
%         acc_semi_supervised(pp,rr) = semi_supervised{pp,rr}.accuracy(yp,y_test);
%         
%     end
% end
%% 
% We can plot the performance curve for this model against the fully supervised 
% case.

figure

macc = mean(acc_subset,2);
sacc = std(acc_subset,[],2);
plot(breaks./max(breaks),macc,'k')
hold on
plot(breaks./max(breaks),macc+2*sacc,'k--')
plot(breaks./max(breaks),macc-2*sacc,'k--')
% 
% macc = mean(acc_semi_supervised,2);
% sacc = std(acc_semi_supervised,[],2);
% plot(breaks./max(breaks),macc,'b')
% hold on
% plot(breaks./max(breaks),macc+2*sacc,'b--')
% plot(breaks./max(breaks),macc-2*sacc,'b--')
% yline(acc_supervised)
xlabel('Number of Labels')
ylabel('Accuracy')
axis tight
ylim([0 1])

figure
plot(acc_subset,'k')
hold on
% plot(acc_semi_supervised,'b')
%% Active DP

nrepeats = 100;

prior.alpha = 10;

active = cell(nrepeats,1);
acc_active = NaN(nrepeats,1);
opts.backsamp_window = 200;
opts.backsamp_loops = 5;
opts.shuf = false;


parfor rr = 1:nrepeats
    
    active{rr} = DPGMM_active(x,y,prior,opts);
    active{rr} = active{rr}.initialise();
    
%     yp = active{rr}.predict_map(x_test);
    yp = active{rr}.getKeys(active{rr}.label);
    
    acc_active(rr) = active{rr}.accuracy(yp,y_test);

end

rr = nrepeats;

figure
plot(acc_subset,'k')
hold on
% plot(acc_semi_supervised,'b')
% scatter(sum(active{rr}.fixed)./length(y)*100,sum(yp==y_test)./length(yp),20,'r','filled')
scatter(cellfun(@(a) sum(a.fixed)./length(y)*100,active),acc_active,20,'r','filled')

figure
plot(x(active{rr}.fixed==false,1),x(active{rr}.fixed==false,2),'k.')
hold on
plot(x(active{rr}.fixed==true,1),x(active{rr}.fixed==true,2),'bx')
%% 
save('results/Z24_active')

