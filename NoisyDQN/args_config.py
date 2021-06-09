"""
	This file contains the arguments to parse at command line.
	File main.py will call get_args, which then the arguments
	will be returned.
"""
import argparse
import torch

def get_args():
	"""
		Description:
		Parses arguments at command line.

		Parameters:
			None

		Return:
			args - the arguments parsed
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--algo", default="DQN")
	parser.add_argument("--env_name", default="CartPole-v0")
	parser.add_argument("--train_eps", type=int, default=200)
	parser.add_argument("--test_eps", type=int, default=50)
	parser.add_argument("--double_DQN", type=bool, default=True)
	parser.add_argument("--target_network_update_frequency", type=int, default=4)

	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=0.0001)
	parser.add_argument('--gamma', type=float, default=0.95, metavar='G', help='discount factor(default:0.99)')
	parser.add_argument("--epsilon_start", type=float, default=1)
	parser.add_argument("--epsilon_end", type=float, default=0.01)
	parser.add_argument("--epsilon_decay", type=float, default=500)
	parser.add_argument("--replay_buffer_capacity", type=int, default=10000)
	parser.add_argument('--hidden_dim', type=int, default=256)
	parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 543)')
	parser.add_argument('--render', action='store_false', help='render the environment')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
						help='interval between training status logs (default: 10)')
	parser.add_argument('--train', dest='train', action='store_true', default=True)

	if torch.cuda.is_available():
		parser.add_argument("--device", type=str, default="cuda")
	else:
		parser.add_argument("--device", type=str, default="cpu")

	args = parser.parse_args()

	return args


