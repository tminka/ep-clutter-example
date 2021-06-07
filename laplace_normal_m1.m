function [e,m,h,k,run] = laplace_normal_m1(prior, data, v1, p2, w)

[d,n] = size(data);
q = ones(2,n)/2;
%q = rand(2,n);
old_m = inf;
iv0 = inv(get_cov(prior));

run.e = [];
run.m = [];
run.flops = [];

niters = 100;
for iter = 1:niters
  % M step
  nj = sum(q(1,:));
	addflops(n-1);
  xj = (data*q(1,:)')/nj;
	addflops(flops_mul(data, q(1,:)') + flops_div);
  vm = inv(nj/v1 + iv0);
	addflops(2*flops_div + 1);
  % this assumes m0 = 0
  m = (vm*nj/v1)*xj;
	addflops(2+flops_div);

  % E step
  diff = data - repmat(m, 1, n);
	addflops(d*n);
  q(1,:) = -0.5*(diff.^2)/v1 -0.5*log(2*pi*v1) + log(1-w);
	addflops(3*d*n);
  q(2,:) = p2 + log(w);
  z = logsumexp(q,1);
	addflops(flops_logsumexp(q,1));
  q = q - repmat(z, rows(q), 1);
  q = exp(q);
	addflops(2*n*(1 + flops_exp));
  
  flops_here = flops;
  g = row_sum(q(1,:).*q(2,:).*(diff.^2));
  h = iv0 + sum(q(1,:)) - g;
  k = sum(z) + logProb(prior, m);
  run.e(iter) = k + 0.5*log(2*pi/h);
  run.m(iter) = m;
  run.flops(iter) = flops;
  flops(flops_here);

  if abs(m - old_m) < 1e-4
    break
  end
  
  old_m = m;
end
if iter == niters
  warning('not enough iters')
end
e = run.e(iter);
