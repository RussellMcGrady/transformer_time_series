"""
code-ish example of how to use the inference function to do validation
during training. 

The validation loop can be used as-is for model testing as well.

NB! You cannot use this script as is. This is merely an example to show the overall idea - 
not something you can copy paste and expect to work. For instance, see "sandbox.py" 
for example of how to instantiate model and generate dataloaders.

If you have never before trained a PyTorch neural network, I suggest you look
at some of PyTorch's beginner-level tutorials.
"""
import torch
import util.inference as inference
import util.utils as utils
import lib.transformer_timeseries as tst
import numpy as np
import util.dataset as ds

from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler


PLOT_BIAS = True
PLOT_PREDICT = True
# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparams
target_col_name = "FCR_N_PriceEUR"
timestamp_col = "timestamp"
test_size = 0.1
batch_size = 64

## Params
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
output_sequence_length = 48 # supposing you're forecasting 48 hours ahead
enc_seq_len = 153 # supposing you want the model to base its forecasts on the previous 7 days of data
window_size = enc_seq_len + output_sequence_length # used to slice data into sub-sequences
step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len
batch_first = False
forecast_window = 48

# Define input variables 
exogenous_vars = [] # should contain strings. Each string must correspond to a column name
input_variables = [target_col_name] + exogenous_vars

# Read data
data = utils.read_data(timestamp_col_name=timestamp_col)

# Get test data from dataset
test_data = data[-(round(len(data)*test_size)):]

# Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc. 
# Should be test data indices only
test_indices = utils.get_indices_input_target(
    num_obs=round(len(data)*test_size),
    input_len=window_size,
    step_size=window_size,
    forecast_horizon=0,
    target_len=output_sequence_length
)

# looks like normalizing input values curtial for the model
scaler = MinMaxScaler(feature_range=(-1, 1))
# scaler = StandardScaler()
# Recover the original values
# original_data = scaler.inverse_transform(scaled_data)
series = test_data[input_variables].values
amplitude = scaler.fit_transform(series)

# Making instance of custom dataset class
test_data = ds.TransformerDataset(
    data=torch.tensor(amplitude).float(),
    indices=test_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=output_sequence_length,
    target_seq_len=output_sequence_length
    )

# Making dataloader
test_data = DataLoader(test_data, batch_size)

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

# Initialize the model with the same architecture and initialization as when it was saved
model = tst.TimeSeriesTransformer(
    input_size=len(input_variables),
    dec_seq_len=enc_seq_len,
    batch_first=batch_first,
    num_predicted_features=1
    ).to(device)

# Define the file path, same as the forecast_window
PATH = 'model/model_{}_{}_e50_lr6.pth'.format(enc_seq_len, output_sequence_length)

# Load the saved state dictionary into the model
model.load_state_dict(torch.load(PATH))
# Load the state dict into the model
# state_dict  = torch.load(PATH, map_location=torch.device('cpu'))
# model.load_state_dict(state_dict)

loss_fn = torch.nn.HuberLoss().to(device)
# loss_fn = torch.nn.MSELoss().to(device)

# Set the model to evaluation mode
model.eval()
print_output = torch.tensor([]).to(device)

# Iterate over all (x,y) pairs in validation dataloader
with torch.no_grad():

    for step, batch in enumerate(test_data):
        total_loss = 0.
        output = torch.Tensor(0)    
        truth = torch.Tensor(0)
        
        src, trg, trg_y = batch
        src, trg, trg_y = src.to(device), trg.to(device), trg_y.unsqueeze(2).to(device)

        # Permute from shape [batch size, seq len, num features] to [seq len, batch size, num features]
        if batch_first == False:

            # shape_before = src.shape
            src = src.permute(1, 0, 2)
            # print("src shape changed from {} to {}".format(shape_before, src.shape))
            trg = trg.permute(1, 0, 2)
            trg_y = trg_y.permute(1, 0, 2)

        # inference on the length of the output window
        prediction = model(
            src=src,
            tgt=trg,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )

        total_loss += loss_fn(prediction, trg_y).item()
        output = torch.cat((output, torch.tensor(scaler.inverse_transform(prediction.squeeze().detach().cpu())).unsqueeze(-1).view(-1).cpu()), 0)
        truth = torch.cat((truth, torch.tensor(scaler.inverse_transform(trg_y.squeeze().detach().cpu())).unsqueeze(-1).contiguous().view(-1).cpu()), 0)
        print_output = torch.cat((print_output, prediction.permute(1, 0, 2)))

    if PLOT_BIAS == True:
        utils.plot(output, truth, step)
    
    if PLOT_PREDICT == True:
        if batch_first == False:
            batch_size = src.size()[1]
        else:
            batch_size = src.size()[0]
        # inference one by one
        forecast = inference.run_encoder_decoder_inference(
            model=model, 
            src=src, 
            forecast_window=forecast_window,
            batch_size=batch_size,
            device=device,
            batch_first=batch_first
            ) # predict forecast_window steps
        # recover scaled data
        src = torch.tensor(scaler.inverse_transform(src[:,-1,].detach().cpu()))
        forecast = torch.tensor(scaler.inverse_transform(forecast[:,-1,].detach().cpu()))
        utils.predict_future(src[:,-1,].unsqueeze(-1), forecast[:,-1,].unsqueeze(-1))
            
    print(scaler.inverse_transform(print_output.squeeze().detach().cpu().numpy()))

