function plot_accuracy_vs_cost(results, color)

algs = fieldnames(results);

% evidence
figure(1),clf
legtxt = {};
for a = 1:length(algs)
	r = results.(algs{a});
	if isfield(r,'e') && isfield(r,'run')
		if isfield(r.run,'e') && isfield(r.run,'flops')
			acc = abs(exp(r.run.e) - exp(results.exact.e));
			h=loglog(r.run.flops,acc,color.(algs{a}));hold on;
			legtxt{end+1} = algs{a};
			%set(h,'linewidth',2);
		end
	end
end
legend(legtxt);
%mobile_text(legtxt)
title('Evidence')
xlabel('FLOPS')
ylabel('Error')
set(gcf,'PaperPosition',[0.25 2.5 4 4])
axis_pct
%axis([1e2 1e6 1e-27 1e-24])
% print -dps clutter_ex3_evidence.ps

% mean
figure(2),clf
legtxt = {};
for a = 1:length(algs)
	r = results.(algs{a});
	if isfield(r,'run') && isfield(r.run,'flops')
		acc = abs(r.run.m - results.exact.m);
		h=loglog(r.run.flops,acc,color.(algs{a}));hold on;
		legtxt{end+1} = algs{a};
		%set(h,'linewidth',2);
	end
end
legend(legtxt);

%results.adf.acc = abs(results.adf.m - results.exact.m);
%  results.ep.acc = abs(results.ep.run.mw - results.exact.m);
  % results.vb.acc = abs(results.vb.run.m - results.exact.m);
  % results.laplace.acc = abs(results.laplace.run.m - results.exact.m);
  % results.brute.acc = abs(results.brute.run.m - results.exact.m);
  % results.gibbs.acc = abs(results.gibbs.m - results.exact.m);
  % loglog(results.ep.run.flops, results.ep.acc, 'x-', ...
  %     results.laplace.run.flops, results.laplace.acc, '--', ...
	% 		results.vb.run.flops, results.vb.acc, '-', ...
  %     results.brute.run.flops, results.brute.acc, '-', ...
	% 		results.gibbs.run.flops, results.gibbs.acc, '-')
  %legend('EP','Laplace','VB','Importance','Gibbs',3)
%axis([1e2 1e6 1e-2 1e1])
axis_pct
  %mobile_text('EP','Laplace','VB','Importance','Gibbs')
  title('Posterior mean')
  xlabel('FLOPS')
  ylabel('Error')
  set(gcf,'PaperPosition',[0.25 2.5 4 4])
  % print -dps clutter_ex3_mean.ps
end
