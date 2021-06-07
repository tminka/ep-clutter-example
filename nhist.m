function h = nhist(data, bins, scale, varargin)

if nargin < 3
	scale = 1;
end
if nargin < 2
  bins = 10;
end

[n,x] = hist(data, bins);
if isscalar(bins)
	width = range(data)/bins;
	n = n / length(data) / width;
else
	% bin i ranges from (x(i-1)+x(i))/2 to (x(i)+x(i+1))/2
	% bin width = (x(i+1)-x(i-1))/2
	bins = bins(:)';
	bins2 = [-inf bins inf];
	width = (bins2(3:end) - bins2(1:end-2))/2;
	n = (n / length(data)) ./ width;
end
n = n * scale;
h = bar(x, n, 'hist', varargin{:});
if nargout == 0
	clear h
end
