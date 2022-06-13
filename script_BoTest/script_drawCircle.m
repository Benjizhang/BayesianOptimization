% plot a cirle
clc; clear; close all;

% x = 1:0.1:5;
% y = 1:0.1:5;
% z = target(x,y);
% plot3(x,y,z)

target(9,1)
target(0,3)

function z = target(x,y)
    z = -power(x,2)-power((y-1),2)+4;
end

