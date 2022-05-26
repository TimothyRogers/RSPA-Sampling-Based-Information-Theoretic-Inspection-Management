

clear 
close all
clc
%%

set(0,'DefaultTextInterpreter','latex')
fig_pos = [1,1,8,4.5];


fontsizer = @(ss) set(findall(gcf,'-property','FontSize'),'FontSize',ss);

load('./results/Z24_active.mat')

%%

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

cmap2 = [252,251,253;242,240,247;239,237,245;218,218,235;203,201,226;188,189,220;158,154,200;128,125,186;117,107,177;106,81,163;84,39,143;74,20,134;63,0,125]./255;
%cmap2 = brewermap(25,'purples');
		
%%
np = 10;
figure('Units','Inches','Position',[1 1 16 9])

% subplot(np,1,1)
for ii = 1:50
%     plot([1:length(y);1:length(y)],
% scatter(1:length(y),1.2*ones(1,length(y)),50,cmap(y,:),'filled')
% axis off
%     xline(ii,'Color',cmap(y(ii),:),'LineWidth',1.5)
end
hold on
% subplot(np,1,2:np)
plot(cell2mat(cellfun(@(a)a.query_prob,active,'UniformOutput',false)'),'Color',[0 0 0 0.005])
hold on
plot(mean(cell2mat(cellfun(@(a)a.query_prob,active,'UniformOutput',false)'),2),'Color',[0 0 0],'LineWidth',2)
ylim([0,1])
xlim([0,length(y)])
xlabel('Data Point')
ylabel('Query Probability')
fontsizer(24)
% subplot(np,1,1)
for ii = 1:length(y)
% scatter(1:length(y),1.2*ones(1,length(y)),50,cmap(y,:),'filled')
% axis off
xline(ii,'Color',cmap(y(ii),:),'LineWidth',1.5)
end
fontsizer(24)
printout('./outputs/z24/query_probabilities_labels')

%%


figure('Units','Inches','Position',fig_pos)
for kk = 1:length(unique(y))
    scatter(x(y==kk,1),x(y==kk,2),12,'k','filled','MarkerFaceColor',cmap(kk,:), 'MarkerFaceAlpha',0.5, 'MarkerEdgeAlpha',0.5)
%     scatter(x_test(y_test==kk,1),x_test(y_test==kk,2),2,cmap(kk,:),'Marker','d')
    hold on
    xlabel('$X_1$')
    ylabel('$X_2$')
end
fontsizer(24)
printout('./outputs/z24/training_data')

%%


figure('Units','Inches','Position',[1 1 16 9])
for kk = 1:length(unique(y))
    scatter(x(y==kk,1),x(y==kk,2),12,'k','filled','MarkerFaceColor',cmap2(kk+5,:),'MarkerEdgeColor',[0.6 0.6 0.6], 'MarkerEdgeAlpha',0.5)
    hold on
    xlabel('$X_1$')
    ylabel('$X_2$')
end
plot_clusters(mu_prior(1:2),Sig_prior(1:2,1:2))
fontsizer(24)
printout('./outputs/z24/prior')

%%

[mu_post,Sig_post] = supervised.MAP();
figure('Units','Inches','Position',fig_pos)
for kk = 1:length(unique(y))
    scatter(x(y==kk,1),x(y==kk,2),12,'k','filled','MarkerFaceColor',cmap(kk,:), 'MarkerFaceAlpha',0.5, 'MarkerEdgeAlpha',0.5)
    hold on
    xlabel('$X_1$')
    ylabel('$X_2$')
    plot_clusters(mu_post(kk,1:2),Sig_post(1:2,1:2,kk),cmap(kk,:),'LineWidth',1.5)
end
% plot_clusters(mu_prior,Sig_prior)
fontsizer(24)
printout('./outputs/z24/fully_supervised')

%%

[mu_post,Sig_post] = unsupervised.MAP();
figure('Units','Inches','Position',fig_pos)
for kk = 1:length(unique(y))
    scatter(x(y==kk,1),x(y==kk,2),12,'k','filled','MarkerFaceColor',cmap(kk,:), 'MarkerFaceAlpha',0.5, 'MarkerEdgeAlpha',0.5)
    hold on
end
for kk = 1:length(unsupervised.cluster)
    plot_clusters(mu_post(kk,1:2),Sig_post(1:2,1:2,kk),cmap(kk,:))
end
xlabel('$X_1$')
ylabel('$X_2$')
fontsizer(24)
printout('./outputs/z24/fully_unsupervised')

%%

[mu_post,Sig_post] = active{1}.MAP();
figure('Units','Inches','Position',fig_pos)
for kk = 1:length(unique(y))
    scatter(x(y==kk & active{1}.fixed==0,1),x(y==kk & active{1}.fixed==0,2),2,'k','filled')
    hold on
    scatter(x(y==kk & active{1}.fixed==1,1),x(y==kk & active{1}.fixed==1,2),15,cmap(kk,:),'filled')
    hold on
end
jj = 1;
for kk =  [1,2,5]
     plot_clusters(mu_post(kk,1:2),Sig_post(1:2,1:2,kk),cmap(jj,:))
     jj = jj+1;
end
for kk =  [3,4]
     plot_clusters(mu_post(kk,1:2),Sig_post(1:2,1:2,kk),[0 0 0 0.2])
     jj = jj+1;
end
xlabel('$X_1$')
ylabel('$X_2$')
fontsizer(24)
printout('./outputs/z24/active_clusters')

%%

% Example semi-supervised
inds = randperm(size(x,1));
labels = NaN(size(y));
uinds = inds(1:sum(active{1}.fixed==1));
labels(uinds) = y(uinds);
semi_example = DPGMM_semi_supervised(x,labels,prior,opts);
semi_example = semi_example.initialise();
semi_example = semi_example.gibbs_inference();
[mu_post,Sig_post] = semi_example.MAP();
ninds = setdiff(1:length(y),inds(1:sum(active{1}.fixed==1)));
figure('Units','Inches','Position',[1 1 16 9])
binds = zeros(length(y)); binds(uinds) = true;
for kk = 1:length(unique(y))
    scatter(x(ninds,1),x(ninds,2),2,'k','filled')
    scatter(x(binds & y == kk,1),x(binds & y == kk,2),15,cmap(kk,:),'filled')
    hold on
    xlabel('$X_1$')
    ylabel('$X_2$')
    plot_clusters(mu_post(kk,:),Sig_post(:,:,kk),cmap(kk,:))
end
% plot_clusters(mu_prior,Sig_prior)
fontsizer(24)
printout(sprintf('./outputs/z24/semi_supervised_%i_datapoints',sum(active{1}.fixed)))

%%

% figure('Units','Inches','Position',[1 1 16 9])
% hold on
% % 
% % for kk = 1:nrepeats
% %     scatter(breaks./max(breaks)*100,acc_subset(:,kk),5,cmap(2,:),'filled','MarkerEdgeAlpha',0,'MarkerFaceAlpha',0.1)
% % end
% % % plot(acc_subset,'.','Color',[0 0 0 0.1])
% % hold on
% % h1 = plot(breaks./max(breaks)*100,nanmean(acc_subset,2),'Color',cmap(2,:), 'LineWidth',2);
% 
% % plot(acc_semi_supervised,'.','Color',[0 0 1 0.1])
% for kk = 1:nrepeats
%     scatter(breaks./max(breaks)*100,acc_semi_supervised(:,kk),5,cmap(1,:),'filled','MarkerEdgeAlpha',0,'MarkerFaceAlpha',0.1)
% end
% hold on
% h2 = plot(breaks./max(breaks)*100,nanmean(acc_semi_supervised,2),'Color',cmap(1,:), 'LineWidth',2);
% % plot(breaks,acc_semi_supervised,cmap(1,:))
% % scatter(sum(active{rr}.fixed)./length(y)*100,sum(yp==y_test)./length(yp),20,'r','filled')
% scatter(cellfun(@(a) sum(a.fixed)./length(y)*100,active),acc_active,20,cmap(3,:),'filled','MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0)
% % 
% h3 = scatter(mean(cellfun(@(a) sum(a.fixed)./length(y)*100,active)),mean(acc_active),20,cmap(3,:),'filled','MarkerEdgeColor','k');
% % h3 = plot(NaN);
% ylim([0.5,1])
% xlim([0,100])
% 
% h4 = yline(acc_supervised,'LineWidth',2);
% 
% xlabel('Percentage of Queried Points')
% ylabel('Test Accuracy')
% % legend([h1,h2,h3],{'Subset','Semi-Supervised','Active'},'location','southeast')
% legend([h2,h3,h4],{'Semi-Supervised','Active','Fully Supervised'},'location','southeast')
% 
% fontsizer(24)
% printout('./outputs/z24/performance curve')



%%

figure('Units','Inches','Position',fig_pos)

for kk = 1:3
    inds = find(y==kk);
    plot(repmat(inds',2,1),repmat([0;1],1,length(inds)),'Color',[cmap(kk,:),1],'LineWidth',0.01)
    hold on
end

plot(cell2mat(cellfun(@(a)a.query_prob,active,'UniformOutput',false)'),'Color',[0 0 0 0.005])
hold on
plot(mean(cell2mat(cellfun(@(a)a.query_prob,active,'UniformOutput',false)'),2),'Color',[0 0 0],'LineWidth',2)
ylim([0,1])
xlim([0,length(y)])
xlabel('Data Point')
ylabel('Query Probability')
fontsizer(24)
printout('./outputs/z24/query_probabilities')

%%

figure('Units','Inches','Position',fig_pos)
subplot(10,1,2:9)
plot(cell2mat(cellfun(@(a)a.query_prob,active,'UniformOutput',false)'),'Color',[0 0 0 0.005])
hold on
plot(mean(cell2mat(cellfun(@(a)a.query_prob,active,'UniformOutput',false)'),2),'Color',[0 0 0],'LineWidth',2)
ylim([0,1])
xlim([0,length(y)])
xlabel('Data Point')
ylabel('Query Probability')
subplot(10,1,1)
scatter(1:length(y),zeros(length(y),1),15,cmap(y,:),'MarkerFaceAlpha',0.5, 'MarkerEdgeAlpha',0.5)
axis off
linkaxes(get(gcf,'children'),'x')
fontsizer(24)
printout('./outputs/z24/query_probabilities_above')


%%
% figure('Units','Inches','Position',[1 1 16 9])
% % plot(cell2mat(cellfun(@(a)a.query_prob,active_ord,'UniformOutput',false)'),'Color',[0 0 0 0.005])
% hold on
% plot(mean(cell2mat(cellfun(@(a)a.query_prob,active_ord','UniformOutput',false)'),2),'Color',[0 0 0 0.4],'LineWidth',0.75)
% for kk = 1:6
%     xline((kk-1)*335,'Color',cmap(kk,:),'LineWidth',4)
%     hold on
% end
% 
% ylim([0,1])
% xlim([0,length(y)])
% xlabel('Data Point')
% ylabel('Query Probability')
% 
% fontsizer(24)
% printout('./outputs/z24/query_probabilities_ordered')


%%

figure('Units','Inches','Position',fig_pos)
scatter(x(active{rr}.fixed==false,1),x(active{rr}.fixed==false,2),2,cmap(2,:),'filled')
hold on
scatter(x(active{rr}.fixed==true,1),x(active{rr}.fixed==true,2),10,cmap(1,:),'filled')
legend('Unqueried','Queried','Fontsize',20,'Location','southeast')
xlabel('$X_1$')
ylabel('$X_2$')
fontsizer(24)
printout('./outputs/z24/queried_points')


