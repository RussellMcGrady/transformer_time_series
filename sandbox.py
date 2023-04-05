"""
Showing how to use the model with some time series data.

NB! This is not a full training loop. You have to write the training loop yourself. 

I.e. this code is just a starting point to show you how to initialize the model and provide its inputs

If you do not know how to train a PyTorch model, it is too soon for you to dive into transformers imo :) 

You're better off starting off with some simpler architectures, e.g. a simple feed forward network, in order to learn the basics
"""

import util.dataset as ds
import util.utils as utils
import torch
import datetime
import time
import lib.transformer_timeseries as tst
import numpy as np
import math

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import MinMaxScaler, StandardScaler


torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparams
test_size = 0.1
batch_size = 64
target_col_name = "FCR_N_PriceEUR"
timestamp_col = "timestamp"
# Only use data from this date and onwards
cutoff_date = datetime.datetime(2017, 1, 1) 

## Params
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
dec_seq_len = 92 # length of input given to decoder
enc_seq_len = 153 # length of input given to encoder
output_sequence_length = 48 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = enc_seq_len + output_sequence_length # used to slice data into sub-sequences
step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len
batch_first = False

# Define input variables 
exogenous_vars = [] # should contain strings. Each string must correspond to a column name
input_variables = [target_col_name] + exogenous_vars
target_idx = 0 # index position of target in batched trg_y

input_size = len(input_variables)

# Read data
data = utils.read_data(timestamp_col_name=timestamp_col)

# Remove test data from dataset
training_data = data[:-(round(len(data)*test_size))]

# Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc. 
# Should be training data indices only
training_indices = utils.get_indices_entire_sequence(
    data=training_data, 
    window_size=window_size, 
    step_size=step_size)

# looks like normalizing input values curtial for the model
scaler = MinMaxScaler(feature_range=(-1, 1))
# scaler = StandardScaler()
# Recover the original values
# original_data = scaler.inverse_transform(scaled_data)
series = training_data[input_variables].values
amplitude = scaler.fit_transform(series)

# Making instance of custom dataset class
training_data = ds.TransformerDataset(
    data=torch.tensor(amplitude).float(),
    indices=training_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=output_sequence_length
    )

# Making dataloader
training_data = DataLoader(training_data, batch_size)

model = tst.TimeSeriesTransformer(
    input_size=len(input_variables),
    dec_seq_len=enc_seq_len,
    batch_first=batch_first,
    num_predicted_features=1
    ).to(device)


# Make src mask for decoder with size:
# [batch_size*n_heads, output_sequence_length, enc_seq_len]
src_mask = utils.generate_square_subsequent_mask(
    dim1=output_sequence_length,
    dim2=enc_seq_len
    ).to(device)

# Make tgt mask for decoder with size:
# [batch_size*n_heads, output_sequence_length, output_sequence_length]
tgt_mask = utils.generate_square_subsequent_mask( 
    dim1=output_sequence_length,
    dim2=output_sequence_length
    ).to(device)

loss_fn = torch.nn.HuberLoss().to(device)
# loss_fn = torch.nn.MSELoss().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

# Define the warm-up schedule
num_epochs = 50
# total_steps = len(training_data) * num_epochs
# Create the scheduler
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=num_epochs)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)


for epoch in range(num_epochs):
    model.train() # Turn on the train mode \o/
    start_time = time.time()
    total_loss = 0.

    for step, batch in enumerate(training_data):
        optimizer.zero_grad()
        
        src, trg, trg_y = batch
        src, trg, trg_y = src.to(device), trg.to(device), trg_y.unsqueeze(2).to(device) # feature size = 1
    
        # Permute from shape [batch size, seq len, num features] to [seq len, batch size, num features]
        if batch_first == False:
            # shape_before = src.shape
            src = src.permute(1, 0, 2)
            # print("src shape changed from {} to {}".format(shape_before, src.shape))
            trg = trg.permute(1, 0, 2)
            trg_y = trg_y.permute(1, 0, 2)

        output = model(
            src=src,
            tgt=trg,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        # output = output.permute(1, 0, 2).squeeze()
        # print(f'output:', output.size())
    
        loss = loss_fn(output, trg_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        log_interval = int(len(training_data) / batch_size / 1)
        if step % (10*log_interval) == 0 and step > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | step {:3d} | '
                    'lr {:02.8f} | {:5.2f} ms | '
                    'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, step, scheduler.get_last_lr()[0], # get_lr()
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        
    if epoch == num_epochs-1:
        print('hidden embeddings of epoch {}: {}'.format(epoch, output))
            
    scheduler.step()

    if (epoch+1) % 10 == 0:
      # Save the model
      torch.save(model.state_dict(), 'model/model_{}_{}.pth'.format(enc_seq_len, output_sequence_length))
      # model.load_state_dict(torch.load('model.pth'))