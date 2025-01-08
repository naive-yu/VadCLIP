import torch
import sys

from utils.tools import get_prompt_text
sys.path.append('.')
from model import CLIPVAD
from torchstat import stat
from torchinfo import summary
from thop import profile
from ucf_option import parser

# 测试模式
args = parser.parse_args()
cuda_index = 0
batch_size = 1
visual_length = 256
image_dim = 512
device = 'cuda:0'
model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window,
				args.prompt_prefix, args.prompt_postfix, device).cuda(cuda_index)
model.eval()
padding_mask = torch.zeros(batch_size, visual_length, dtype=torch.bool)  # 示例掩码
# label_map = dict({'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 'Assault': 'Assault', 'Burglary': 'Burglary',
# 					'Explosion': 'Explosion', 'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery',
# 					'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 'Vandalism': 'Vandalism'})
# prompt_text = get_prompt_text(label_map)
prompt_text = torch.randn((14, 512)).cuda(cuda_index)
input_tensor = torch.randn(batch_size, visual_length, image_dim).cuda(cuda_index)
lengths = torch.full((1,), 256).cuda(cuda_index)
summary(model, input_data=[input_tensor, padding_mask, prompt_text, lengths])

Flops, params = profile(model, (input_tensor, padding_mask, prompt_text, lengths))  # macs
print('Flops: % .4fG' % (Flops / 1000000000))  # 计算量


# model.py修改：
# text_features_ori = self.encode_textprompt(text)
# text_features_ori = text