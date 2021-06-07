function [e,mw,vw] = ep_normal_full(prior, x, v1, p2, w)
% Input:
% prior is a normal_density
% x is data (cols)
% v1 is variance of the component
% p2 is row vector of logprobs for x under clutter model

[d,n] = size(x);
a = zeros(1,n);
m = zeros(d,n);
v = cell(1,n);
for i = 1:n
  v{i} = 1e8*eye(d);
end

mp = get_mean(prior);
vp = get_cov(prior);

ivw = inv(vp);
mw = vp\mp;
for i = 1:n
  ivw = ivw + inv(v{i});
  mw = mw + v{i}\m(:,i);
end
vw = inv(ivw);
mw = vw*mw;

niters = 20;
for iter = 1:niters
  old_m = m;
  for i = 1:n
    v0 = inv(inv(vw) - inv(v{i}));
    m0 = v0*(inv(vw)*mw - inv(v{i})*m(:,i));

    p1 = mvnormpdfln(x(:,i),m0,[],v0+v1);
    t0 = (1-w)*exp(p1) + w*exp(p2(i));
    %t0 = exp(logsumexp([log(1-w)+p1; log(w)+p2(i)], 1));
    r = (1-w)*exp(p1)/t0;
    
    iv1 = inv(v0+v1);
    v{i} = inv(r*iv1 - r*(1-r)*iv1*(x(:,i)-m0)*(x(:,i)-m0)'*iv1) - v0;
    dm = r*iv1*(x(:,i) - m0);
    m(i) = m0 + (v0 + v{i})*dm;
    a(i) = log(t0) + 0.5*logdet(v0/v{i} + eye(d)) + 0.5*(m(:,i)-m0)'*inv(v0+v{i})*(m(:,i)-m0);
  
    mw = m0 + v0*dm;
    vw = inv(inv(v0) + inv(v{i}));
  end

  s = mp'*inv(vp)*mp - mw'*inv(vw)*mw;
  for i = 1:n
    s = s + m(:,i)'*inv(v{i})*m(:,i);
  end
  e(iter) = sum(a) + 0.5*logdet(vw) -0.5*trace(s) - 0.5*logdet(vp);
  
  if max(abs(old_m - m)) < 1e-4
    break
  end
end
if iter == niters
  warning('ep_normal: not enough iters')
end
figure(2)
plot(e)
e = e(iter);
