% mixture of two gaussians with one mean free

addpath('lightspeed')
addpath('density')

v = {1 10};
d = 1;
w = 1/2;
prior = {normal_density(zeros(d,1), 10^2*eye(d)) ...
         normal_density(zeros(d,1), 0*eye(d))};
density1 = normal_density(2*ones(d,1), v{1}*eye(d));
density2 = normal_density(zeros(d,1), v{2}*eye(d));
density1 = set_prior(density1, prior{1});
density2 = set_prior(density2, prior{2});
mix = mixture_density([1-w w], density1, density2);

% EP performs better for large n, while sampling remains constant
n = 20;
data = sample(mix, n);

% Data similar that used in thesis figure 3-5 and UAI paper figure 1
%data = [2.002106149625671   0.555176328097228  -4.318471009764671   0.047668280197101   2.122678698276165   3.682754179364856  -0.594880267196699  4.162340754786174   1.078357560413008   0.870576889541305   1.460813128155038   5.981936936269046   0.689118570918456   1.470221311235092   4.211148921430878   1.567643102656376   1.615396948864076   2.311104566891609  -3.148049184358392   2.364878513651955];

% Data used in MinkaMaterialsTutorial.pdf
%data = [1.21370270241168 1.10529117370787 4.73164396953918 2.41107915462951 0.742888745069693 1.57956932375192 -1.61524585094746 0.693138031089869 2.81211102159522 2.10953735238992];

% Data where ADF is very bad, but EP is good
%data = [-10.555437907388010   1.671996845772058   0.578967657612911   4.134986290076893   1.024124843829640   3.803842413458685  -0.437947966385572  -0.091226237935124   3.798543238149592   0.728956803840643   2.997484605436716   1.203810919560722   2.181930016321307   7.672034014674877   1.289607619585368  -3.269906319838870   1.793421676496840   1.375221587889087   5.604194442202764   1.997569681299016];

% Data with an interesting bimodal posterior
%data = [4.281774722174348   2.052631489375457  -0.878944628368981   2.309540310018265  -5.298342129691209   2.037148270721817   1.775633850249606  -1.702763610942165   5.542760118591918  -0.930341425561346  -1.260938397285787   1.508842559215285  -7.453133327831017   1.711021534461259  -2.024795989907581  -0.946427232699952   3.988644698976740  -0.091721184710906  -4.640430510608936  -3.022482866484175];

% Data that is slightly bimodal, causes problems for EP
%data = [1.385308700895383   0.872426341152790   3.089798280465828   0.765533799122779   3.661201597710336   5.526658460306225  -0.121849608779596   0.722983894738821  -2.511238750244174   3.150787325508161   0.754464437730618  -3.895682966067144   2.613466351692358   3.305219744214705  -1.114907228432438   4.463286942278552 -10.272966598288951   3.933067047270920   1.499151921163576   1.605030748949069];

% Infer.NET 2-point test
%data = [0.1 2.3];
% Infer.NET 1-point test
%data = [2.3];

% multimodal
%data = [-3.7 -1.7 0.8 1.3 4.2];
% good example for EP
%data = [-0.5 0 1.5 2 2.3];
% EP cannot converge on this data, even with damping (for w=1/3)
%data = [-1 0 2 3];

% Permute the data
%data = data(randperm(n));

n = length(data);
if isvector(data)
	figure(3)
	hist(data)
  %set(gca,'xtick',[],'ytick',[])
  set(gcf,'PaperPosition',[0.25 2.5 2 2])
else
	figure(3)
  plot(data(1,:), data(2,:), '.')
  set(gca,'xtick',[],'ytick',[])
  set(gcf,'PaperPosition',[0.25 2.5 2 2])
  xlabel('y1')
  ylabel('y2')
  title('Typical data')
end
if 0
  % kernel estimate
  obj = train(kernel_density, data);
end

results = struct;

% ADF and EP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p2 = logProb(density2, data);
flops(0);
adf = struct;
[adf.e,adf.m,adf.v] = adf_normal_sphere(prior{1}, data, v{1}, p2, w);
adf.flops = flops;
results.ADF = adf;

disp('EP')
flops(0);
ep = struct;
[ep.e,ep.m,ep.v,ep.run] = ep_normal_sphere(prior{1}, data, v{1}, p2, w);
%[ep.e,ep.m,ep.v] = sep_normal_sphere(prior{1}, data, v{1}, p2, w);
ep.flops = flops;
results.EP = ep;

%disp('SEP')
%[adf.e,adf.m,adf.v] = sep_normal_sphere(prior{1}, data, v{1}, p2, w);
%ep = adf;
%return

% Reverse EM for normal means %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if 1
  disp('VB')
  flops(0);
  vb = struct;
  [vb.q,vb.run] = vb_normal_m_train(prior, v, data, w);
  vb.e = vb_normal_m_bound(prior, v, data, vb.q);
  vb.m = vb.run.m(end);
  vb.v = vb.run.v(end);
  results.VB = vb;
end

disp('Laplace')
flops(0);
laplace = struct;
[laplace.e,laplace.m,laplace.h,laplace.k,laplace.run] = laplace_normal_m1(prior{1}, data, v{1}, p2, w);
laplace.v = 1/laplace.h;
results.Laplace = laplace;

if 1
disp('Importance sampling')
flops(0);
[importance.e,importance.m,importance.run] = importance_normal_m1(prior{1}, data, v{1}, p2, w);
importance.v = sum(importance.run.weight.*(importance.run.sample - importance.m).^2)/sum(importance.run.weight);
results.Importance = importance;

disp('Gibbs sampling')
flops(0);
[gibbs.m,gibbs.run] = gibbs_normal_m1(prior{1}, data, v{1}, p2, w, 1000);
%gibbs.v = mean((gibbs.run.sample - gibbs.m).^2);
gibbs.v = mean((gibbs.run.mm - gibbs.m).^2) + mean(gibbs.run.vm);
q = cat(2,gibbs.run.qs{:});
q = reshape(q(1,:),cols(q)/4,4);
gibbs.q = mean(q);
results.Gibbs = gibbs;
end

if 0
% ML fit %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1)
clf
fit = train(mix, data);
disp(fit)
mbr = classify(fit, data);
if 0
  % force hard assignment (cf Dom)
  hard = mbr;
  hard(find(hard < 0.5)) = 0;
  hard(find(hard > 0.5)) = 1;
end

% Laplace's method %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c = get_components(fit);
laplace.m = get_mean(c{1});
laplace.k = sum(logProb(fit, data)) + logProb(prior{1}, laplace.m);
g = data - repmat(laplace.m, 1, cols(data));
s = row_sum(mbr(1,:).*mbr(2,:).*(g.^2));
laplace.h = 1/get_cov(prior{1}) + sum(mbr(1,:)) - s;
end

% Plot the fits %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ms = linspace(-40,40,2000);
inc = ms(2)-ms(1);
exact = struct;
exact.f = zeros(length(ms),1);
em_bound = zeros(length(ms),1);
hard_bound = zeros(length(ms), 1);
vb.f = repmat(-Inf,length(ms),1);
for i = 1:length(ms)
  obj = set_component(mix, 1, set_mean(density1, ms(i)));
  exact.f(i) = sum(logProb(obj, data)) + logProb(prior{1}, ms(i));
  %em_bound(i) = sum(logProb_bound(obj, data, mbr)) + logProb(prior{1}, ms(i));
  %hard_bound(i) = sum(logProb_bound(obj, data, hard)) + logProb(prior{1}, ms(i));
  vb.f(i) = sum(logProb_bound(obj, data, vb.q)) + logProb(prior{1}, ms(i));
  %vb.f(i) = vb_normal_m_curve(prior, v, data, vb.q, {ms(i) 0});
end
if 0
  clf
  axis([-10 10 0 500])
  draw(density2,'b')
  draw(density1)
  axis auto
  set(gca,'ytick',[])
  set(gca,'xtick',[0 2],'xticklabel',['0'; 'x'])
  set(gcf,'PaperPosition',[0.25 2.5 3.5 1.5])
end

exact.ehat = logsumexp(exact.f,1)+log(inc);
exact.e = exact.ehat;
exact.m = sum(ms'.*exp(exact.f - exact.e + log(inc)));
%exact.v = sum((ms.*ms)'.*exp(exact))/exp(logSum(exact)) - exact_m^2;
exact.v = sum((ms - exact.m).^2*exp(exact.f - exact.e + log(inc)));
results.exact = exact;
best.f = exact.e - 0.5*log(2*pi*exact.v) -0.5*((ms - exact.m).^2)'/exact.v;
results.bestGaussian = best;

if isfield(results,'VB')
  results.VB = vb;
  results.VB.ehat = logsumexp(vb.f,1)+log(inc);
end

laplace.f = laplace.k -0.5*laplace.h*((ms - laplace.m).^2)';
laplace.ehat = logsumexp(laplace.f,1)+log(inc);
results.Laplace = laplace;
adf.f = adf.e -0.5*log(2*pi*adf.v) -0.5*((ms - adf.m).^2)'/adf.v;
results.ADF.ehat = logsumexp(adf.f,1)+log(inc);
ep.f = ep.e -0.5*log(2*pi*ep.v) -0.5*((ms - ep.m).^2)'/ep.v;
ep.ehat = logsumexp(ep.f,1)+log(inc);
results.EP = ep;
importance.f = importance.e -0.5*log(2*pi*importance.v) -0.5*((ms - importance.m).^2)'/importance.v;
importance.ehat = logsumexp(importance.f,1)+log(inc);
results.Importance = importance;

% Gibbs sampling approximates the posterior using complete cond density avging
p = zeros(size(ms));
for i = 1:cols(gibbs.run.sample)
  p = p + mvnormpdf(ms, gibbs.run.mm(i), [], gibbs.run.vm(i));
end
gibbs.f = log(p/cols(gibbs.run.sample));
results.Gibbs = gibbs;

% Best Gaussian bound %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if 0
c = get_components(fit);
y = get_mean(c{1});
k = sum(logProb(fit, data)) + logProb(prior{1}, y);
b = 5;
kby = fmins('normal_bound_fcn', [k;b;y], [], [], ms, exact);
[k,b,y] = deal(kby(1), kby(2), kby(3));
ep_approx = k -1/2*b*((y-ms).^2)';

min(exact - ep_approx)
end

% for n = 10, vb_bound is very close to the best gaussian bound

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the posteriors

color = struct;
color.exact = 'm';
color.EP = 'b-x';
%color.ADF = 'g--';
color.VB = 'r';
color.Laplace = 'g';
%color.bestGaussian = 'm';
color.Importance = 'c';
color.Gibbs = 'm';

figure(3);clf
if isvector(data)
	nhist(data,10,exp(results.exact.e),color.exact)
	hold on
end
algs = fieldnames(results);
legtxt = {'Data'};
for a = 1:length(algs)
  r = results.(algs{a});
  if isfield(r,'f') && isfield(r,'e') && isfield(color,algs{a})
    figure(3);h=plot(ms, exp(r.f), color.(algs{a}));hold on;
    legtxt{end+1} = algs{a};
    set(h,'linewidth',2);
  end
end

figure(3)
xlabel('\theta')
ylabel('p(\theta,D)')
hold off
legend(legtxt);
ax = axis;
axis([min(data) max(data) ax(3) ax(4)])
set(gca,'ytick',[])
set(gcf,'PaperPosition',[0.25 2.5 4 4])
% print -dps clutter_ex3_posterior.ps

if 0
	% Focus on the Gibbs posterior
  figure(3)
  plot(ms, exp(exact.f - exact.e), '-', ms, gibbs.f, '--')
  xlabel('\theta')
  ylabel('p(\theta | D)')
	ax = axis;
	axis([-1 5 ax(3) ax(4)])
  set(gcf,'PaperPosition',[0.25 2.5 4 4])
  legend('Exact','Gibbs')
end
if 0
  figure(1)
  rn = cols(gibbs.run.ms);
  p = zeros(1,rn);
  for i = 1:rn
    m = gibbs.run.ms(i);
    obj = set_component(mix, 1, set_mean(density1, m));
    p(i) = sum(logProb(obj, data)) + logProb(prior{1}, m);
  end
  plot(ms, exp(exact), '-', gibbs.run.ms, exp(p), 'o')
  xlabel('theta')
  ylabel('p(theta, D)')
  set(gca,'xtick',[],'ytick',[])
  set(gcf,'PaperPosition',[0.25 2.5 4 4])
  legend('Exact','Gibbs')
end

% Compute the integral approximations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Evidence:')
%disp(['EM = ' num2str(logSum(em_bound)+log(inc))]);
%disp(['(actually ' num2str(vb_normal_m_bound(prior, v, data, mbr)) ')'])
for a = 1:length(algs)
  r = results.(algs{a});
  if isfield(r,'e')
    if ~isfield(r,'ehat')
      r.ehat = nan;
    end
    fprintf('  %-7s = %g (actually %g)\n', algs{a}, r.ehat, r.e);
  end
end

if 1
	% Plot the error by algorithm
  figure(1),clf
  acc = [];
  legtxt = {};
  for a = 1:length(algs)
    if strcmp(algs{a},'exact')
      continue
    end
    r = results.(algs{a});
    if isfield(r,'e')
      acc(end+1) = abs(exp(r.e) - exp(exact.e));
      legtxt{end+1} = algs{a};
    end
  end
  xtick = 1:length(acc);
  semilogy(xtick,acc,'o')
  axis_pct
  set(gca,'xtick',xtick,'xticklabel',legtxt);
  ylabel('Error')
  title('Evidence')
end

% Compare the posterior means %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp('Posterior mean:')
% this is highly dependent on knowing the variances
naive_m = mean(data)/(1-w);
fprintf('  %-7s = %g\n', 'Naive', naive_m);
for a = 1:length(algs)
  r = results.(algs{a});
  if isfield(r,'m')
    fprintf('  %-7s = %g\n', algs{a}, r.m);
  end
end

if 1
	% Plot the error by algorithm
  figure(2),clf
  acc = [];
  legtxt = {};
  for a = 1:length(algs)
    if strcmp(algs{a},'exact')
      continue
    end
    r = results.(algs{a});
    if isfield(r,'m')
      acc(end+1) = abs(r.m - exact.m);
      legtxt{end+1} = algs{a};
    end
  end
  xtick = 1:length(acc);
  semilogy(xtick,acc,'o')
  axis_pct
  set(gca,'xtick',xtick,'xticklabel',legtxt);
  ylabel('Error')
  title('Mean')
end

% Compare the posterior variances %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp('Posterior variance:')
for a = 1:length(algs)
  r = results.(algs{a});
  if isfield(r,'v')
    fprintf('  %-7s = %g\n', algs{a}, r.v);
  end
end

plot_accuracy_vs_cost(results, color)

if 0
  [ep.e,ep.m,ep.v,run] = ep_normal_sphere(prior{1}, data, v{1}, p2, w);
  [y,i] = sort(run.v);
  [y,i] = sort(data);
  i = fliplr(i);
end

% Three ADF fits
% task3 is nice for this
if 0
  figure(2)
  clf
  plot(ms, exp(exact), '--')
  hold on
  p = perms(1:n);
  sv = 0;
  sm = 0;
  for iter = 1:rows(p)
    %i = randperm(n);
    i = p(i,:);
    data = data(i);
    p2 = p2(i);
    [adf.e,adf.m,adf.v] = adf_normal_sphere(prior{1}, data, v{1}, p2, w);
    adf_approx = adf.e -0.5*log(2*pi*adf.v) -0.5*((ms - adf.m).^2)'/adf.v;
    figure(2)
    plot(ms, exp(adf_approx), 'g-')

    % compute the grand mean of the mixture
    sv = sv + exp(adf.e);
    sm = sm + exp(adf.e)*adf.m;
  end
  sm = inv(sv)*sm;
  fprintf('error = %g\n', abs(exact_m - sm));
  
  figure(2)
  hold off
  ax = axis;
  axis([-1 5 0 ax(4)])
  legend('Exact', 'ADF')
  xlabel('\theta')
  ylabel('p(\theta,D)')
  set(gca,'ytick',[])
  % use small size to make it look good in figures
  set(gcf,'PaperPosition',[0.25 2.5 3 3])
end
