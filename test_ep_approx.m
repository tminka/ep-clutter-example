% show t versus tilde{t}

w = 0.5;
x = 2;
p2 = normpdfln(x, 0, [], 10);

theta = linspace(-5,10,200);
f = (1-w)*normpdf(theta, x, [], 1) + w*exp(p2);

mts = [0 3];
vts = [0.1 1 10];
vts = 0.1;
clf
for i = 1:length(vts)
  vt = vts(i);
  for j = 1:length(mts)
    % when vt is large, mt doesn't matter
    mt = mts(j);

    q = normpdf(theta, mt, [], vt);
    p1 = normpdfln(x, mt, [], vt+1);
    z = (1-w)*exp(p1) + w*exp(p2);
    r = (1-w)*exp(p1)/z;
    
    vi = inv(r/(vt+1) - r*(1-r)*(x-mt)^2/(vt+1)^2) - vt;
    mi = mt + (vi + vt)*r*(x-mt)/(vt+1);
    %s = z/sqrt(2*pi*vi)*exp(-gauss_logProb(mi, mt, vi+vt));
    s = z*sqrt(1 + vt/vi)*exp(0.5/(vi+vt)*(mi - mt)^2);
    
    g = s*exp(-0.5*(theta-mi).^2/vi);
    subplot(length(vts),length(mts),(i-1)*length(mts)+j);
    h = plot(theta, g, '-r', theta, f, '-b', theta, q, 'g');
    set(h,'LineWidth',2);
    title(['m_\theta = ' num2str(mt) ', v_\theta = ' num2str(vt)])
    axis([-5 10 0 0.3])
    if i == length(vts)
      xlabel('\theta')
    end
    set(gca,'ytick',[])
    if i == 1 && j == 1
      legend('approx', 'factor', 'context')
    end
  end
end
if 0
	q = normpdf(theta, 0, [], 0.1);
	f = 0.5*normpdf(theta, 2, [], 1) + 0.5*normpdf(2, 0, [], 10);
	inc = theta(2)-theta(1);
	plot(theta, q, theta, f, theta, q.*f./(sum(q.*f)*inc))
	axis_pct
	legend('Prior','Likelihood','Posterior')
	%axis([-5 10 0 0.3])
	z = sum(q.*f);
	mnew = sum(theta.*q.*f)/z
	vnew = sum((theta-mnew).^2.*q.*f)/z
end
