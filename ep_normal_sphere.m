function [e,mw,vw,run] = ep_normal_sphere(prior, x, v1, p2, w)
% EP_NORMAL_SPHERE   Expectation Propagation for the clutter problem.
%
% ep_normal_sphere(prior, x, v1, p2, w)
% prior is a normal_density
% x is data (cols)
% v1 is variance of the component
% p2 is row vector of logprobs for x under clutter model

[d,n] = size(x);
a = zeros(1,n);
m = zeros(d,n);
v = Inf*ones(1,n);

mp = get_mean(prior);
vp = get_cov(prior);
vp = vp(1,1);

ivw = 1/vp;
mw = mp;

run.e = [];
run.m = [];
run.flops = [];

step = 1;
niters = 100/step;
for iter = 1:niters
  old_m = m;
  old_v = v;
  for i = 1:n
    v0 = 1/(ivw - 1/v(i));
		addflops(1+2*flops_div);
    if v0 < 0
      fprintf('skipping %d on iter %d\n', i, iter);
      continue
      %warning('v0 < 0')
      %iter = niters;
      %break
    end
    %m0 = v0*(sum(m./v) + mp/vp - m(:,i)/v(i));
    %m0 = v0*(ivw*mw - inv(v(i))*m(:,i));
    m0 = mw + (v0/v(i))*(mw - m(:,i));
		addflops(3+flops_div);
    
    p1 = mvnormpdfln(x(:,i),m0,[],(v0+v1)*eye(d));
		addflops(flops_normpdfln(x(:,i),m0,[],(v0+v1)*eye(d)));
    if ~isreal(p1)
      fprintf('EP: normalizer is imaginary after %d iters\n', iter)
      e = nan;
      mw = nan;
      vw = nan;
      return
    end
    t0 = (1-w)*exp(p1) + w*exp(p2(i));
		addflops(4+2*flops_exp);
    %t0 = exp(logSum(log(1-w)+p1, log(w)+p2(i)));
    r = (1-w)*exp(p1)/t0;
		addflops(flops_div);
    
    iv1 = inv(v0+v1);
    dif = x(:,i)-m0;
    dv = r*iv1 - r*(1-r)*iv1*(dif'*dif)*iv1/d;
    v(i) = inv(dv) - v0;
		addflops(11+3*flops_div);
    % force positive v(i) for convergence
    if v(i) < 0
      %error('v(i) < 0')
      % a large number, not infinity
      %v(i) = 1e8;
    end
    dm = r*iv1*dif;
    m(:,i) = m0 + (v0 + v(i))*dm;
		addflops(5);

    if step ~= 1
      b = m(:,i)/v(i);
      old_b = old_m(:,i)/old_v(i);
      v(i) = 1/((1-step)*1/old_v(i) + step*1/v(i));
      b = (1-step)*old_b + step*b;
      m(:,i) = v(i)*b;
    end
    a(i) = log(t0) + d/2*log(v0/v(i) + 1) ...
				+ 0.5*sum((m(:,i)-m0).^2)/(v0+v(i));
    
    %ivw = inv(v0 - v0*dv*v0);
    % second formula is valid for any v(i)
    ivw = inv(v0) + inv(v(i));
    % these are equivalent when using second formula for ivw above
    %mw = m0 + v0*dm;
    mw = inv(ivw)*(inv(v0)*m0 + inv(v(i))*m(:,i));
		addflops(4+3*flops_div);
    
    if 0 & (v0 < 10)
      % check derivative constraint
      v0
      inc = 0.1;
      vs = 0.1:0.1:10;
      for j = 1:length(vs)
				p1 = gauss_logProb(x(:,i),m0,(vs(j)+v1)*eye(d));
				exact(j) = logSum(log(1-w)+p1, log(w)+p2(i));
				approx(j) = a(i) + d/2*log(v(i)) - d/2*log(vs(j)+v(i)) ...
						- 0.5*(m(:,i)-m0)'*(m(:,i)-m0)/(vs(j)+v(i));
      end
      figure(2)
      plot(vs, exact, vs, approx)
      %g = -0.5*r/(v0+v1) + 0.5*r/(v0+v1)^2*(x(i)-m0)^2;
      %plot(vs, gradient(exact)/inc, vs, gradient(approx)/inc, vs, g)
      pause
    end
  end
  
  vw = inv(ivw);
  s = (mp'*mp)/vp - (mw'*mw)*ivw;
  for i = 1:n
    s = s + (m(:,i)'*m(:,i))/v(i);
  end
  run.e(iter) = sum(a) + d/2*log(vw) -1/2*s - d/2*log(vp);
  run.m(iter) = mw;
  run.flops(iter) = flops;

  activity = max(abs(old_m - m));
  activity = max(activity, max(abs(old_v - v)));
  %fprintf('activity = %g\n', activity);
  if activity < 1e-4
    break
  end
end
if iter == niters
  warning('ep_normal: not enough iters')
end
e = run.e(iter);

run.v = v;
