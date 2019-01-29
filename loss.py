import torch.nn as nn

class fashionLoss(nn.Module):
    def __init__(self, attr_index):
        super(fashionLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.attr_index = attr_index

    def forward(self, output, label):
        output_attr0 = output[:,0:8]
        output_attr1 = output[:,8:14]
        output_attr2 = output[:,14:20]
        output_attr3 = output[:,20:29]
        output_attr4 = output[:,29:34]
        output_attr5 = output[:,34:39]
        output_attr6 = output[:,39:44]
        output_attr7 = output[:,44:54]
        
        result = 0

        if self.attr_index == 0:
            result = self.loss(output_attr0, label.squeeze() )
        elif self.attr_index == 1:
            result = self.loss(output_attr1, label.squeeze() )
        elif self.attr_index == 2:
            result = self.loss(output_attr2, label.squeeze() )
        elif self.attr_index == 3:
            result = self.loss(output_attr3, label.squeeze() )
        elif self.attr_index == 4:
            result = self.loss(output_attr4, label.squeeze() )
        elif self.attr_index == 5:
            result = self.loss(output_attr5, label.squeeze() )
        elif self.attr_index == 6:
            result = self.loss(output_attr6, label.squeeze() )
        elif self.attr_index == 7:
            result = self.loss(output_attr7, label.squeeze() )

        return result
