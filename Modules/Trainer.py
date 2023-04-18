import os
import time
import torch
import numpy as np
from tqdm import tqdm
from .Evaluation import Evaluation

class Trainer(object):
    def __init__(self, model, trainGenerator, validGenerator, optim, lossFunc, topN, resultDir):
                
        self.topN = topN
        self.model = model
        self.optim = optim
        self.lossFunc = lossFunc
        self.resultDir = resultDir
        self.device = model.device
        self.evalutor = Evaluation(self.model, self.lossFunc, k=topN)
        
        self.trainGenerator = trainGenerator
        self.validGenerator = validGenerator
        
    def train(self, nEpochs=10):
        for epoch in range(nEpochs):
            st = time.time()
            print('Start Epoch #', epoch)
            
            trainLoss, events = self.trainEpoch(epoch)
            # validLoss, recall, mrr = self.evalutor.evalute(self.validGenerator)
            validLoss, recall, mrr = None, None, None
            epoch_duration = time.time() - st
            print(f'epoch:{epoch} loss: {trainLoss:.6f} {epoch_duration:.2f} s {np.sum(events)/epoch_duration:.2f} e/s {len(events)/epoch_duration:.2f} mb/s')
            self.saveModel(epoch, validLoss, trainLoss, recall, mrr) 

    def trainEpoch(self, epoch):
        losses = []
        events = []
        self.model.train()
        batchSize = float(self.trainGenerator.batchSize)
        hidden = self.model.initHidden(batchSize)
        negative = self.model.negative
        
        for _ , (input, target, finishedMask, validMask) in tqdm(enumerate(self.trainGenerator),
                                                                total=self.trainGenerator.totalIters,
                                                                miniters=1000, position=0, leave=True):
            input = input.to(self.device)
            target = target.to(self.device)            
            hidden = self.model.resetHidden(hidden, finishedMask, validMask)
            logit, hidden = self.model(input, hidden, target)

            # output sampling
            if(negative):
                loss = self.lossFunc(logit)
            else:
                loss = self.lossFunc(logit, target)
                
            loss = (float(len(input)) / batchSize) * loss
            events.append(len(input))
            
            if(~np.isnan(loss.item())):
                losses.append(loss.item())
                loss.backward()
                self.optim.step() 
                self.optim.zero_grad()
            
        meanLoss = np.mean(losses)
        return meanLoss, events
    
    def saveModel(self, epoch, validLoss, trainLoss, recall, mrr):
        checkPoints = {
              'model': self.model,
              'epoch': epoch,
              'optim': self.optim,
              'validLoss': validLoss,
              'trainLoss': trainLoss,
              'recall': recall,
              'mrr': mrr
        }
        modelName = os.path.join(self.resultDir, "model_{0:05d}.pt".format(epoch))
        torch.save(checkPoints, modelName)
        print("Save model as %s" % modelName)
