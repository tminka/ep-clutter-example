function [m,run] = gibbs_normal_m1(prior, data, v1, p2, w, nsamples)

if nargin < 6
  nsamples = 1000;
end

[d,n] = size(data);
iv0 = inv(get_cov(prior));

run.qs = cell(1,nsamples);
run.mm = zeros(d,nsamples);
run.vm = zeros(d,nsamples);
run.sample = zeros(d,nsamples);
run.ns = zeros(1,nsamples);
run.flops = zeros(1,nsamples);

for outer = 1:1
  q = zeros(2,n);
  r = (rand(1,n) < 0.5)+1;
  for i = 1:n
    q(r(i),i) = 1;
  end
  for iter = 1:nsamples
    % M step
    nj = sum(q(1,:));
		addflops(flops_row_sum(q(1,:)));
    if nj > 0
      xj = (data*q(1,:)')/nj;
			addflops(flops_mul(data, q(1,:)') + flops_div);
      vm = inv(nj/v1 + iv0);
      mm = (vm*nj/v1)*xj;
			addflops(3*flops_div + 3);
    else
      vm = get_cov(prior);
      mm = get_mean(prior);
    end
    m = randnorm(1, mm, [], vm);
		addflops(flops_randnorm(1, mm, [], vm));

    run.qs{iter} = q;
    run.mm(:,iter) = mm;
    run.vm(:,iter) = vm;
    run.sample(:,iter) = m;
    run.ns(:,iter) = nj;
    run.flops(iter) = flops;
    
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
    % make q binary by sampling
    for i = 1:n
      r = sample(q(:,i));
      q(:,i) = zeros(2,1);
      q(r,i) = 1;
    end
		addflops(2*n);
  end
end
run.m = cumsum(run.mm)./(1:length(run.mm));
m = mean(run.mm,2);
