import torch
import numpy as np
from torch.autograd import Variable
from utils.mahalanobis_lib import get_Mahalanobis_score


def model_forward(model, data):
    return model(data)

def iterate_data_msp(data_loader, model):
    confs = []
    cls = []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits, _ = model_forward(model, x)
            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data)
            cls.extend(y)
        if b % 100 == 0:
            print('{} batches processed'.format(b))
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()

def iterate_data_msp_custom(data_loader, model, targets):
    confs = []
    cls = []
    targets = torch.tensor(targets).cuda()
    targets = targets.unsqueeze(0)
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits, _ = model_forward(model, x)
            softmax_output = m(logits)

            sim = softmax_output - targets
            conf, _ = torch.max(sim, dim=-1)
            confs.extend(conf.data)
            cls.extend(y)
        if b % 100 == 0:
            print('{} batches processed'.format(b))
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()



def iterate_data_odin(data_loader, model, epsilon, temper):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    cls = []
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)
        outputs, _ = model_forward(model, x)
        cls.extend(y)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
        outputs, _ = model_forward(model, Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))
        if b % 100 == 0:
            print('{} batches processed'.format(b))
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()

def iterate_data_odin_custom(data_loader, model, epsilon, temper, targets, mode='linear'):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    cls = []
    targets = np.expand_dims(targets, axis=0)
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)
        outputs, _ = model_forward(model, x)
        softmax_output = m(outputs)
        softmax_output = softmax_output.data.cpu()
        softmax_output = softmax_output.numpy()
        cls.extend(y)
        sim = -softmax_output * targets
        sim = sim.sum(axis=1) / (np.linalg.norm(softmax_output, axis=-1) * np.linalg.norm(targets, axis=-1))
        sim = np.expand_dims(sim, axis=1)
        sim = sim + 1

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
        outputs, _ = model_forward(model, Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        
        nnOutputs = sim * nnOutputs
        confs.extend(np.max(nnOutputs, axis=1))
        if b % 100 == 0:
            print('{} batches processed'.format(b))
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()

def iterate_data_energy(data_loader, model, temper):
    confs = []
    cls = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits, _ = model_forward(model, x)
            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data)
            cls.extend(y)
            if b % 100 == 0:
                print('{} batches processed'.format(b))
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()

def iterate_data_energy_custom(data_loader, model, temper, targets, mode='linear'):
    confs = []
    cls = []
    targets = torch.tensor(targets).cuda()
    targets = targets.unsqueeze(0)
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits, _ = model_forward(model, x)
            conf = temper * torch.logsumexp(logits / temper, dim=1) #(batch)

            softmax_output = m(logits)
            sim = -softmax_output * targets
            sim = sim.sum(1) / (torch.norm(softmax_output, dim=1) * torch.norm(targets, dim=1))
            sim = sim + 1
            conf = conf * sim
            confs.extend(conf.data)
            cls.extend(y)
            if b % 100 == 0:
                print('{} batches processed'.format(b))
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()

def iterate_data_mahalanobis(data_loader, model, num_classes, sample_mean, precision,
                             num_output, magnitude, regressor):
    confs = []
    cls = []
    for b, (x, y) in enumerate(data_loader):
        if b % 100 == 0:
            print('{} batches processed'.format(b))
        x = x.cuda()
        Mahalanobis_scores = get_Mahalanobis_score(x, model, num_classes, sample_mean, precision, num_output, magnitude)
        scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]
        confs.extend(scores)
        cls.extend(y)
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()

def iterate_data_gradnorm_custom(data_loader, model, temperature, num_classes,targets):
    confs = []
    cls = []
    targets = torch.tensor(targets).cuda()
    targets = targets.unsqueeze(0)
    with torch.no_grad():
        for b, (x, y) in enumerate(data_loader):
            if b % 10 == 0:
                print('{} batches processed'.format(b))
            inputs = Variable(x.cuda(), requires_grad=False)
            outputs, features = model_forward(model, inputs)
            U = torch.norm(features, p=1, dim=1)
            out_softmax = torch.nn.functional.softmax(outputs, dim=1)
            V = torch.norm((targets - out_softmax), p=1, dim=1)
            S = U * V / 2048 / num_classes
            confs.extend(S)
            cls.extend(y)
    return torch.tensor(confs).cuda(), torch.tensor(cls).cuda()

def iterate_data_gradnorm(data_loader, model, temperature, num_classes):
    confs = []
    labels = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        if b % 100 == 0:
            print('{} batches processed'.format(b))
        inputs = Variable(x.cuda(), requires_grad=True)
        model.zero_grad()
        outputs, _ = model_forward(model, inputs)
        targets = torch.ones((inputs.shape[0], num_classes)).cuda()
        outputs = outputs / temperature

        loss = torch.sum(torch.mean(-targets * logsoftmax(outputs), dim=-1))

        loss.backward()

        layer_grad = model.fc.weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad))
        confs.append(layer_grad_norm)
        label = y.clone().detach()
        labels.append(label)
    return torch.tensor(confs).cuda(), torch.tensor(labels).cuda()
