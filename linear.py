import torch 
from torch.autograd import Variable 
from torch.nn import functional as F 
from torch.utils.data import DataLoader
import sys
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(precision=1)


from models import LinearRegression
from dataset import PowerDataset


model = LinearRegression()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

model = model.float() 

# Training data loader
dataloader = DataLoader(
    PowerDataset('Data_set_no_formula.xlsx'),
    batch_size=6,
    shuffle=True,
)

# Test data loader
val_dataloader = DataLoader(
	PowerDataset(path='Data_set_no_formula.xlsx', mode="test"),
    batch_size=5,
    shuffle=True,
    # num_workers=1,
)


#### Trainging ####
running_loss = 0.0
for epoch in range(1):
	for i, batch in enumerate(dataloader):
		y_val = batch['y_val']
		x_val = batch['x_val']
		optimizer.zero_grad()

		y_pred = model(x_val.float())

		# loss = criterion(y_pred, y_val)
		loss = criterion(y_pred.float(), y_val.float())
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		if i%5 ==4:
			sys.stdout.write(
				"\r[Epoch %d] [Batch %d] [MSELoss loss: %f]"
			    % (
			        epoch,
			        i,
			        loss.item(),
			    )
			)
			running_loss = 0.0
print('\nFinished Training')


#### Testing model and printing absolute difference between real and predicted #### 
length_data = len(val_dataloader)
print(length_data)
model.eval()
for epoch in range(1):
	for i, batch in enumerate(val_dataloader):
		y_val_test = batch['y_val']
		x_val_test = batch['x_val']

		y_pred_test = model(x_val_test.float())
		# print("Real Values: \t",  y_val_test)
		# print("Predicted Values: \t",  y_val_test)
		if i == 0: 
			result = (abs(y_pred_test - y_val_test)).data.numpy()
		print("Absolute Difference:\t", (abs(y_pred_test - y_val_test)).data.numpy())
# print(result.shape)
# new_xticks = ['Tsym ms', 
# 			'PayloadSymbNb ms', 
# 			'Tpacket ms', 
# 			'TTN 30sec/Day Fair Use', 
# 			"Radio/MPU(per cycle period)",	
# 			"lag/Lead MCU (per Cycle Period)",	
# 			"Sensor (Per Cycle period)",	
# 			"Sleep Current (in Cycle period)",	
# 			"Summary (per Tx Cycle)", 
# 			"Avg Battery life in (Years)"]

plt.title("Absolute Difference of Read and Predicted")
plt.xlabel("Predicted Colums (tsym etc)")
plt.ylabel("Absolute Difference")
for data in result: 
	plt.plot(data)
plt.show()
