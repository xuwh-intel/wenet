import torch

# import habana_frameworks.torch.gpu_migration
import habana_frameworks.torch.core as htcore


device = torch.device("hpu")

x = torch.rand([1, 256, 50]).to(device)
y = torch.nn.functional.glu(x, dim=1)
print(x)
print(y)

# # Target are to be un-padded and unbatched (effectively N=1)
# T = 50      # Input sequence length
# C = 20      # Number of classes (including blank)

# # Initialize random batch of input vectors, for *size = (T,C)
# input = torch.randn(T, C).log_softmax(1).detach().requires_grad_()
# input_lengths = torch.tensor(T, dtype=torch.long)

# # Initialize random batch of targets (0 = blank, 1:C = classes)
# target_lengths = torch.randint(low=1, high=T, size=(), dtype=torch.long)
# target = torch.randint(low=1, high=C, size=(target_lengths,), dtype=torch.long)
# ctc_loss = torch.nn.CTCLoss(reduction="sum")
# loss = ctc_loss(input.to(device), target.to(device), input_lengths.to(device), target_lengths.to(device))
# loss.backward()
htcore.mark_step()

# print(input, target, input_lengths, target_lengths)
# print(loss)