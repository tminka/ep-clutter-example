function p = vb_normal_m_bound(prior, v, data, q)
% prior is a cell array of normal prior densities for the means.
% v is a cell array of variances for the components.
% data is a matrix of columns.
% q is a matrix of columns.
% p is a scalar.

J = rows(q);
[d,N] = size(data);

n = row_sum(q);
p = -sum(sum(q.*log(q))) + sum(n.*log(1/J));

% loop components
for j = 1:J
  xj = (data * q(j,:)')/n(j);
  s = vtrans(outer(data,data) * q(j,:)', d);
  s = s - n(j)*xj*xj';
  p = p -(n(j)-1)/2*logdet(2*pi*v{j}) -d/2*log(n(j)) -1/2*trace(s/v{j});
  m0 = get_mean(prior{j});
  v0 = get_cov(prior{j});
  p = p + mvnormpdfln(xj, m0, [], v{j}/n(j) + v0);
end
