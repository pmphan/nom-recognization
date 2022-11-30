from torch import nn

class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        T, b, h = output.size()
        output = output.view(T * b, h)

        output = self.embedding(output)
        output = output.view(T, b, -1)

        return output


class Model(nn.Sequential):
    def __init__(self, NCLASS):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),

            nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),

            nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, stride=1, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 1), padding=(0,1)),
    
            nn.Conv2d(256, 512, 3, stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=(2,2), padding=(0,1)),

            nn.Conv2d(512, 512, 2, stride=(1,2)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 2, stride=(2,2)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.Sequential(
            BiLSTM(512, 256, 256),
            BiLSTM(256, 256, NCLASS),
            # Limit output range
            nn.LogSoftmax(dim=2)
       )

    def forward(self, x):
        out = self.stack(x)
        out = out.squeeze(2)
        # For CTC
        out = out.permute(2, 0, 1)
        out = self.lstm(out)
        return out
