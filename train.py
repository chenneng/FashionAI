from loss import fashionLoss

def equal(attr_index, output, label):
    equalnum = 0
    for i in range(len(output)):
        max_index = 0
        max_outlabel = 0
        output_slice = 0
        if attr_index == 0:
            output_slice = output[:,0:8]
        elif attr_index == 1:
            output_slice = output[:,8:14]
        elif attr_index == 2:
            output_slice = output[:,14:20]
        elif attr_index == 3:
            output_slice = output[:,20:29]
        elif attr_index == 4:
            output_slice = output[:,29:34]
        elif attr_index == 5:
            output_slice = output[:,34:39]
        elif attr_index == 6:
            output_slice = output[:,39:44]
        elif attr_index == 7:
            output_slice = output[:,44:54]
       
        for index, outlabels in enumerate(output_slice[i]):
            if outlabels > max_outlabel:
                max_outlabel = outlabels
                max_index = index

        if max_index == label[i].item():
            equalnum += 1

    return equalnum

def train(attr_index, epoch, model, optimizer, train_loader):
    model.train()
    loss = fashionLoss(attr_index)
    lr = 0
    for para in optimizer.param_groups:
        lr = para['lr']

    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.cuda(),label.cuda()
        optimizer.zero_grad()
        output = model(data)
        trainloss = loss(output, label)
        trainloss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Attribute: {}\tEpoch: {}\t[batch: {}/{}]\tBatch loss: {:.4f}\tlr: {:.5f}'.format(
                attr_index, epoch, batch_idx, len(train_loader), trainloss.item(), lr))


def evaluate(attr_index, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss = fashionLoss(attr_index)

    for data, label in test_loader:
        data, label = data.cuda(),label.cuda()
        output = model(data)
        test_loss += loss(output, label).item()
        correct += equal(attr_index, output, label)

    test_loss /= len(test_loader)
    accuracy = 100. * correct / (len(test_loader) * test_loader.batch_size)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader) * test_loader.batch_size,accuracy))
