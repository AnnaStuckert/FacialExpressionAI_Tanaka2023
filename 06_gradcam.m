%{
@author: tanakayudai

Apply Grad-CAM to CNN
%}

%Read the fine-tuned network
net = coder.loadDeepLearningNetwork("*****.mat");
inputSize = net.Layers(1).InputSize(1:2);
%Resize image size
img = imread("*****.bmp");
img = imresize(img,inputSize);
%Classify the image
[classfn,score] = classify(net,img)
%Compute the Grad-CAM map
map = gradCAM(net,img,classfn);
%Show the Grad-CAM map
imshow(img);
hold on;
imagesc(map,'AlphaData',0.5);
colormap jet
hold off;
title("Grad-CAM");
%Save the Grad-CAM map
output = map;
writematrix(output,['*****.csv']);




