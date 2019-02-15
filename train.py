from loss import fashionLoss
import torch

def equal(outputN, label):
    equalnum = 0

    output_attr1 = outputN[0]
    output_attr2 = outputN[1]
    output_attr3 = outputN[2]
    output_attr4 = outputN[3]

    for i, output in enumerate(output_attr1):
        index = torch.argmax(label[i])
        out_index = 30
        if index >= 0 and index < 8:
            out_index = torch.argmax(output_attr1[i]).item()
        elif index >= 8 and index < 14:
            out_index = torch.argmax(output_attr2[i]).item() + 8
        elif index >= 14 and index < 20:
            out_index = torch.argmax(output_attr3[i]).item() + 14
        elif index >= 20 and index < 29:
            out_index = torch.argmax(output_attr4[i]).item() + 20

        if out_index == index:
            equalnum += 1

    return equalnum

def train(epoch, model, optimizer, train_loader):
    model.train()
    loss = fashionLoss()
    lr = 0
    for para in optimizer.param_groups:
        lr = para['lr']

    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(data)
        #print(output)

        #print(output[0].size())
        #print(output[1].size())
        #print(output[2].size())
        #print(output[3].size())

        trainloss = loss(output, label)
        trainloss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Epoch: {}\t[batch: {}/{}] Batch loss: {:.4f}\tlr: {:.5f}'.format(
                epoch, batch_idx, len(train_loader), trainloss.item(), lr))


def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss = fashionLoss()

    for data, label in test_loader:
        data, label = data.cuda(),label.cuda()
        output = model(data)
        test_loss += loss(output, label).item()
        correct += equal(output, label)

    test_loss /= len(test_loader)
    accuracy = 100. * correct / (len(test_loader) * test_loader.batch_size)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader) * test_loader.batch_size,accuracy))
