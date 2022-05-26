%% Active DP Experiments
% 
% 
% We will compare the active DP to a fully supervised and semi-supervised GMM 
% with random label assignments.

clear 
close all
clc

set(0,'DefaultTextInterpreter','latex')

fontsizer = @(ss) set(findall(gcf,'-property','FontSize'),'FontSize',ss);

%%

load('data.mat')

[y,inds] = sort(y);
x = x(inds,:);

x = x(y~=2,:);
x_test = x_test(y_test~=2,:);
y = y(y~=2);
y_test = y_test(y_test~=2);

y(y>2) = y(y>2)-1;
y_test(y_test>2) = y_test(y_test>2)-1;


X = [x;x_test];
Y = [y;y_test];

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

%% 
% Data is the usual Lawrence drifting MDOF dataset with 6 classes. 2010 datapoints 
% in the test set and 990 in the training set.
% 
% Let's visualise the data

figure('Units','Inches','Position',[1 1 16 9])
for kk = 1:6
    scatter(x(y==kk,1),x(y==kk,2),8,cmap(kk,:),'filled')
%     scatter(x_test(y_test==kk,1),x_test(y_test==kk,2),2,cmap(kk,:),'Marker','d')
    hold on
    xlabel('$X_1$')
    ylabel('$X_2$')
end
fontsizer(24)
printout('./outputs/mdof/training_data')

%% 
% Figrue above shows all the training data as filled dots and test data as diamonds.
%% Specification of the prior
% 
% 
% Using an emiprical Bayes prior based on the data, i.e. all one big Gaussian. 
% We will use this prior in all the experiments
% 
% x_test = (x_test - mean(x))./std(x);
% x = (x-mean(x))./std(x);


prior.m0 = mean(x);
prior.k0 = 1;
prior.n0 = size(x,2);
prior.S0 = (prior.n0+size(prior.m0,2)+2)*cov(x);
prior.S0 = (prior.n0+size(prior.m0,2)+2)*eye(size(x,2));%cov(x);
prior.alpha = 5;%ones(1,length(unique(y)))*length(y)/length(unique(y));

opts.shuf = true;

%% 
% We can specify and plot the prior, it is shown here with the training data.

C = NIW(prior.m0,prior.k0,prior.n0,prior.S0);
[mu_prior,Sig_prior] = C.MAP();

figure('Units','Inches','Position',[1 1 16 9])
for kk = 1:length(unique(y))
    scatter(x(y==kk,1),x(y==kk,2),8,cmap(kk,:),'filled')
    hold on
    xlabel('$X_1$')
    ylabel('$X_2$')
end
plot_clusters(mu_prior,Sig_prior)
fontsizer(24)
printout('./outputs/mdof/prior')

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

figure('Units','Inches','Position',[1 1 16 9])
for kk = 1:length(unique(y))
    scatter(x(y==kk,1),x(y==kk,2),8,cmap(kk,:),'filled')
    hold on
    xlabel('$X_1$')
    ylabel('$X_2$')
    plot_clusters(mu_post(kk,:),Sig_post(:,:,kk),cmap(kk,:))
end
plot_clusters(mu_prior,Sig_prior)
fontsizer(24)
printout('./outputs/mdof/fully_supervised')
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
% 
opts.gibbs_steps = 250;

unsupervised = DPGMM_unsupervised(x,prior,opts);
unsupervised = unsupervised.initialise();
unsupervised = unsupervised.gibbs_inference();
[mu_post,Sig_post] = unsupervised.MAP();

figure('Units','Inches','Position',[1 1 16 9])
for kk = 1:length(unique(y))
    scatter(x(y==kk,1),x(y==kk,2),8,cmap(kk,:),'filled')
    hold on
    xlabel('$X_1$')
    ylabel('$X_2$')
    plot_clusters(mu_post(kk,:),Sig_post(:,:,kk),cmap(kk,:))
end
plot_clusters(mu_prior,Sig_prior)
fontsizer(24)
printout('./outputs/mdof/fully_unsupervised')

yp = unsupervised.predict_map(x_test);
acc_unsupervised = unsupervised.accuracy(yp,y_test);
fprintf('Test Accuracy: %.1f%%\n',acc_unsupervised*100)
%% 
% 
%% Supervised GMM - Random Subsets

nrepeats = 5;
nstep = 50;

subset = cell(nstep,nrepeats);
breaks = floor(linspace(20,length(y),nstep));
acc_subset = NaN(nstep,nrepeats);   

% Example subset
pp =5;
inds = randperm(length(y));
xx = x(inds(1:breaks(pp)),:);
yy = y(inds(1:breaks(pp)));
subset_example = GMM_supervised(xx,yy,prior);
[mu_post,Sig_post] = subset_example.MAP();
figure('Units','Inches','Position',[1 1 16 9])
for kk = 1:length(unique(y))
    scatter(xx(yy==kk,1),xx(yy==kk,2),8,cmap(kk,:),'filled')
    hold on
    xlabel('$X_1$')
    ylabel('$X_2$')
    plot_clusters(mu_post(kk,:),Sig_post(:,:,kk),cmap(kk,:))
end
plot_clusters(mu_prior,Sig_prior)
fontsizer(24)
printout(sprintf('./outputs/mdof/subset_%i_datapoints',breaks(pp)))

% Example semi-supervised
labels = NaN(size(y));
labels(inds(1:breaks(pp))) = yy;
semi_example = DPGMM_semi_supervised(x,labels,prior,opts);
semi_example = semi_example.initialise();
semi_example = semi_example.gibbs_inference();
[mu_post,Sig_post] = semi_example.MAP();
ninds = setdiff(1:length(y),inds(1:breaks(pp)));
figure('Units','Inches','Position',[1 1 16 9])
for kk = 1:length(unique(y))
    scatter(x(ninds,1),x(ninds,2),2,'k','filled')
    scatter(xx(yy==kk,1),xx(yy==kk,2),15,cmap(kk,:),'filled')
    hold on
    xlabel('$X_1$')
    ylabel('$X_2$')
    plot_clusters(mu_post(kk,:),Sig_post(:,:,kk),cmap(kk,:))
end
plot_clusters(mu_prior,Sig_prior)
fontsizer(24)
printout(sprintf('./outputs/mdof/semi_supervised_%i_datapoints',breaks(pp)))

%% Repeated Subsets

for rr = 1:nrepeats
    fprintf('Subset Repeat: %i\n',rr)
    parfor pp = 2:nstep
        
        inds = randperm(length(y));

%         labels = zeros(size(y));
%         labels(inds(1:breaks(pp))) = y(inds(1:breaks(pp)));
        
        subset{pp,rr} = GMM_supervised(x(inds(1:breaks(pp)),:),y(inds(1:breaks(pp))),prior);
        
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


semi_supervised = cell(nstep,nrepeats);
breaks = floor(linspace(0,length(y),nstep));
acc_semi_supervised = NaN(nstep,nrepeats);   
opts.gibbs_steps = 250;

for rr = 1:nrepeats
    parfor pp = 1:nstep
        
        inds = randperm(length(y));
        
        labels = NaN(size(y));
        labels(inds(1:breaks(pp))) = y(inds(1:breaks(pp)));
        
        semi_supervised = DPGMM_semi_supervised(x,labels,prior,opts);
        semi_supervised = semi_supervised.initialise();
        semi_supervised = semi_supervised.gibbs_inference();
        
        yp = semi_supervised.getKeys(semi_supervised.predict_map(x_test));
        
        acc_semi_supervised(pp,rr) = semi_supervised.accuracy(yp,y_test);
        
    end
end
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

macc = mean(acc_semi_supervised,2);
sacc = std(acc_semi_supervised,[],2);
plot(breaks./max(breaks),macc,'b')
hold on
plot(breaks./max(breaks),macc+2*sacc,'b--')
plot(breaks./max(breaks),macc-2*sacc,'b--')
yline(acc_supervised)
xlabel('Number of Labels')
ylabel('Accuracy')
axis tight
ylim([0 1])

figure
plot(acc_subset,'k')
hold on
plot(acc_semi_supervised,'b')
%% Active DP

prior.alpha = 1;

active = cell(nrepeats,1);
acc_active = NaN(nrepeats,1);
opts.backsamp_window = 500;
opts.backsamp_loops = 5;



parfor rr = 1:nrepeats
    
    active{rr} = DPGMM_active(x,y,prior,opts);
    active{rr} = active{rr}.initialise();
    
    yp = active{rr}.predict_map(x_test);
    yp = active{rr}.getKeys(yp);
    
    acc_active(rr) = active{rr}.accuracy(yp,y_test);

end

rr = 1;

figure
plot(breaks/max(breaks)*100,acc_subset,'k')
hold on
plot(breaks/max(breaks)*100,acc_semi_supervised,'b')
% scatter(sum(active{rr}.fixed)./length(y)*100,sum(yp==y_test)./length(yp),20,'r','filled')
scatter(cellfun(@(a) sum(a.fixed)./length(y)*100,active),acc_active,20,'r','filled')

figure
plot(x(active{rr}.fixed==false,1),x(active{rr}.fixed==false,2),'k.')
hold on
plot(x(active{rr}.fixed==true,1),x(active{rr}.fixed==true,2),'bx')

%%

save('mdof_new_heristic_drop_class')

