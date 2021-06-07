function [e,m,run] = importance_normal_m1(prior, data, v1, p2, w, nsamples)
% Importance sampling for the clutter problem

[d,n] = size(data);
v0 = get_cov(prior);
c = sqrt(v0(1,1));
groupsize = 1;

total = 0;
cm = zeros(d,1);
run.e = [];
run.m = [];
run.flops = [];
count = 0;

if nargin < 6
  nsamples = 1000;
end
while(nsamples > 0)
  howmany = min([nsamples groupsize]);
  %disp(['sampling ' num2str(howmany)])
  m = c*randn(d, howmany);
  nsamples = nsamples - howmany;

  % compute likelihood
  diff = data - repmat(m, 1, n);
	addflops(d*n);
  q(1,:) = -0.5*(diff.^2)/v1 -0.5*log(2*pi*v1) + log(1-w);
	addflops(3*d*n);
  q(2,:) = p2 + log(w);
  r = exp(sum(logsumexp(q,1)));
	addflops(flops_logsumexp(q,1) + n-1 + flops_exp);
  
  count = count + 1;
  total = total + r;
  cm = cm + (r/total)*(m - cm);
  run.e(count) = log(total/count);
  run.m(:,count) = cm;
  run.weight(count) = r;
  run.sample(:,count) = m;
  run.flops(count) = flops;
end
e = log(total/count);
m = cm;
