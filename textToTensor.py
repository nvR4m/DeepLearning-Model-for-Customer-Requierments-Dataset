import torch
import pandas as pd
def textToTensor(column):
	max_l = 0
	ts_list = []
	for text in column:
		ts_list.append(torch.ByteTensor(list(bytes(text, 'utf8'))))
		max_l = max(ts_list[-1].size()[0], max_l)
	text_t = torch.zeros((len(ts_list), max_l), dtype=torch.float32)
	for i, ts in enumerate(ts_list):
		text_t[i, 0:ts.size()[0]] = ts
	return text_t