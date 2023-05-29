%{
@author: tanakayudai

Transfer learning with SqueezeNet
%}

% Load dataset
imds = imageDatastore('***','IncludeSubfolders',true, 'LabelSource','foldernames');

% Divide the data into training, validation, and test datasets
[imdsRest, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');
[imdsTrain, imdsValidation] = splitEachLabel(imdsRest, 0.8, 'randomized');

% Load pretrained network
net = squeezenet;
inputSize = net.Layers(1).InputSize;

% Replace final layers
lgraph = layerGraph(net);
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
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

% Freeze initial layers
layers = lgraph.Layers;
connections = lgraph.Connections;
layers(1:4) = freezeWeights(layers(1:4));
lgraph = createLgraphUsingConnections(layers,connections);

% Data augmentation
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation, ...
    'DataAugmentation',imageAugmenter);

% Specify the training options
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

% Train the network
[netTransfer,info] = trainNetwork(augimdsTrain,lgraph,options);

% Save the model
save("***","netTransfer");

% Save the history of accuracy and loss
output = [info.TrainingAccuracy; info.ValidationAccuracy; info.TrainingLoss; info.ValidationLoss], writematrix(output, "***");

% Prepare test data for evaluation
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

% Classify test images
[YPred,scores] = classify(netTransfer,augimdsTest);
YTest = imdsTest.Labels;

% Compute the confusion matrix
confMatrix = confusionmat(YTest, YPred);

% Save the confusion matrix as a CSV file
writematrix(confMatrix, "***");

% Calculate final accuracy on test data
final_test_accuracy = mean(YPred == YTest);

% Save the final results on test data
writematrix(final_test_accuracy, '***');
save('***', 'YPred');
