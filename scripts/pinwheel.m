function [features labels] = pinwheel( radial_std, tangential_std, ...
                                       num_classes, num_per_class, rate)
%
% [features labels] = PINWHEEL( radial_std, tangential_std, num_classes,
%                               num_per_class, rate )
% 
% This function generates a "pinwheel" data set.  It has as many arms as
% classes.  It generates them by taking Gaussian distributions,
% stretching them and then rotating them appropriately.  The centers are
% equidistant around the unit circle.
%
% INPUT:
%   - radial_std:     the standard deviation in the radial direction
%   - tangential_std: the standard deviation in the tangential direction
%   - num_classes:    how many arms and classes to generate
%   - num_per_class:  how many of each class to generate
%   - rate:           how many radians to turn per exp(radius)
%
% OUTPUT:
%   - features: the 2d locations in space
%   - labels:   the actual class labels
%
% Reasonable usage example:
%  >> X = pinwheel(0.3, 0.3, 3, 1000, 0.25);
%  >> plot(X(:,1), X(:,2), '.');
%
% Copyright: Ryan Prescott Adams, 2008
% This is released under the GNU Public License.
% http://www.gnu.org/licenses/gpl-2.0.txt
%

  
  % Find the equidistant angles.
  rads = linspace(0, 2*pi, num_classes+1);
  rads = rads(1:end-1);
  
  features = randn([ num_classes*num_per_class 2 ]) ...
      * diag([tangential_std radial_std]) ...
      + repmat([1 0], [num_classes*num_per_class 1]);
  labels   = vec(repmat([1:num_classes], [num_per_class 1]) ...
                 .* ones([num_per_class num_classes]));
  angles   = rads(labels)' + rate*exp(features(:,1));

  % This would probably be faster if vectorized.
  for i=1:size(angles,1)
    features(i,:) = features(i,:) ...
        * [ cos(angles(i)) -sin(angles(i)) ; ...
            sin(angles(i))  cos(angles(i))];
  end
  
  function zz=vec(xx)
    zz = xx(:);
  end
  
end