function [s,m,v] = adf_normal_sphere(prior, x, v1, p2, w)
% Input:
% prior is a normal_density
% x is data (cols)
% p2 is row vector of logprobs for x under clutter model

[d,n] = size(x);
m = get_mean(prior);
v = get_cov(prior);
v = v(1,1);
s = 0;
for i = 1:n
  p1 = mvnormpdfln(x(:,i),m,[],(v+v1)*eye(d));
  t0 = (1-w)*exp(p1) + w*exp(p2(i));
  s = s + log(t0);
  r = (1-w)*exp(p1)/t0;
  iv1 = inv(v+v1);
  dif = x(:,i)-m;
  new_v = v - v*(r*iv1 - r*(1-r)*iv1*(dif'*dif)*iv1/d)*v;
  m = m + (v*r*iv1)*dif;
  v = new_v;
end
