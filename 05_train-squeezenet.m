%{
@author: tanakayudai

Transfer learning with SqueezeNet
%}

%Load dataset
imds = imageDatastore('*****', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%Divide the data into training and validation datasets
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized')
%Record image names of each datasets
writecell(imdsTrain.Files,'*****.csv');
writecell(imdsValidation.Files,'*****.csv');
%Load pretrained network
net = squeezenet;
net.Layers(1)
inputSize = net.Layers(1).InputSize;
%Replace final layers
lgraph = layerGraph(net);
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer]
numClasses = numel(categories(imdsTrain.Labels));
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])
%Freeze initial layers
layers = lgraph.Layers;
connections = lgraph.Connections;
layers(1:4) = freezeWeights(layers(1:4));
lgraph = createLgraphUsingConnections(layers,connections);
analyzeNetwork(lgraph)
%Data augmentation
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
%Specify the training options
miniBatchSize = 512;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');
%Train the replaced network
[netTransfer,info] = trainNetwork(augimdsTrain,lgraph,options);
%Classify validation images
[YPred,scores] = classify(netTransfer,augimdsValidation);
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)
%save model
save("*****.mat","netTransfer")
%save the history of accuracy and loss
output = [info.TrainingAccuracy;info.ValidationAccuracy;info.TrainingLoss;info.ValidationLoss]';
writematrix(output,"*****.csv");
