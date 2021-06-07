function [q,run] = vb_normal_m_train(prior, v, data, w)
% prior is a cell array of normal prior densities for the means.
% v is a cell array of variances for the components.
% data is a matrix of columns.

if w ~= 0.5
  error('w must be 0.5')
end

J = length(prior);
[d,N] = size(data);
q = ones(J,N)/J;
%q = rand(J,N);
run.e = [];
run.m = [];
run.v = [];
run.flops = [];
niters = 100;
for iter = 1:niters
  old_q = q;
  n = row_sum(q);
	addflops(flops_row_sum(q));
  % loop components
  for j = 1:J
    xj = (data * q(j,:)')/n(j);
		addflops(flops_mul(data, q(j,:)') + flops_div);
    m0 = get_mean(prior{j});
    v0 = get_cov(prior{j});
    if v0 == 0
      vm = v0;
      m = m0;
    else
      vj = v{j}/n(j);
      vm = vj*inv(vj + v0)*v0;
      m = vm*(vj\xj + v0\m0);
			addflops(4*flops_div + 5);
    end
    diff = data - repmat(m, 1, N);
		addflops(d*N);
    q(j,:) = -1/2*col_sum(diff.*(inv(v{j})*diff));
    q(j,:) = q(j,:) -1/2*logdet(2*pi*v{j}) -1/2*trace(vm/v{j});
		addflops(flops_inv(d)+flops_mul(d,d,N)+d*N+flops_col_sum(d,N)+flops_mul(vm,v{j})+d+flops_det(d)+flops_log+1);
    if j == 1
      run.m(iter) = m;
      run.v(iter) = vm;
    end
  end
  q = q - repmat(logsumexp(q,1), rows(q), 1);
  q = exp(q);
	addflops(J*N*(2*flops_exp + 2) + N*flops_log);

  flops_here = flops;
  run.e(iter) = vb_normal_m_bound(prior, v, data, q);
  run.flops(iter) = flops;
  flops(flops_here);

  if norm(q - old_q) < 1e-5
    break
  end
end
if iter == niters
  warning('not enough iters')
end
