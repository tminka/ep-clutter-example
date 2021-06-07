function [s,m,v] = adf_normal_full(prior, x, v1, p2, w)
% Input:
% prior is a normal_density
% x is data (cols)
% p2 is row vector of logprobs for x under clutter model

[d,n] = size(x);
m = get_mean(prior);
v = get_cov(prior);
s = 0;
for i = 1:n
  p1 = mvnormpdfln(x(:,i),m,[],v+v1);
  t0 = (1-w)*exp(p1) + w*exp(p2(i));
  s = s + log(t0);
  r = (1-w)*exp(p1)/t0;
  iv1 = inv(v+v1);
  new_v = v - v*(r*iv1 - r*(1-r)*iv1*(x(:,i)-m)*(x(:,i)-m)'*iv1)*v;
  m = m + r*(v*iv1*(x(:,i) - m));
  v = new_v;
end
