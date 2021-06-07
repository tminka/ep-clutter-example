function p = vb_normal_m_curve(prior, v, data, q, m)
% prior is a cell array of normal prior densities for the means.
% v is a cell array of variances for the components.
% data is a matrix of columns.
% q is a matrix of columns.
% m is a cell array of means for the components.
% p is a scalar.

J = rows(q);
[d,N] = size(data);

n = row_sum(q);
p = 0;
for j = 1:J
  if 1
    xj = (data * q(j,:)')/n(j);
    s = vtrans(outer(data,data) * q(j,:)', d);
    s = s - n(j)*xj*xj';
    p = p +1/2*logdet(2*pi*v{j}/n(j))-n(j)/2*logdet(2*pi*v{j}) ...
	-1/2*trace(s/v{j});
    p = p + mvnormpdfln(xj, m{j}, [], v{j}/n(j));
  else
    p = p + (mvnormpdfln(data, m{j}, [], v{j}) * q(j,:)');
  end
end
 
