from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import cv2
from PIL import Image
from torch.nn import functional as nnf
from tqdm import tqdm
from torch.utils.data import Dataset
import pickle
import sys


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def image_grid(imgs, rows, cols):
    pils = imgs
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def make_images(imgs):
    w, h = imgs[0].size
    grids = []
    for x in range(len(imgs)):
        grid = Image.new('RGB', size=(w,h))
        grid.paste(imgs[x])
        grids.append(grid)
    return grids



def read_video(path, transform=None, frames_num=1):
    frames = []
    cap = cv2.VideoCapture(path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    N = length // 2 # //(frames_num)
    #print(length)
    #counter =

    current_frame = 1
    for i in range(length):

        #frameId = int(round(cap.get(current_frame)))
        #print(current_frame)
        ret, frame = cap.read(current_frame)


        if ret and i==current_frame and len(frames)<frames_num:
            size = 360, 360
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame.thumbnail(size, Image.ANTIALIAS)

            frames.append(frame)
            current_frame += N


        #print(current_frame)
        #cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)


    cap.release()
    #print(frames)
    return frames


class ClipCocoDataset(Dataset):
    
    def __init__(self, data_path: str,  prefix_length= 50, gpt2_type = "gpt2",
                 normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        
        #self.image_ids = [caption["image_id"] for caption in captions_raw]
        
        self.captions = captions_raw
        
        
        self.captions_tokens = []
        self.caption2embedding = []
        max_seq_len = 0
        i=0
        for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64))
                self.caption2embedding.append(self.prefixes[i])
                i+=1
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
                
            # self.max_seq_len = max_seq_len
        #del self.captions_tokens
        #del self.caption2embedding
        #gc.collect()
        #with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
        #        pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
       
    
    
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask
    
    def __len__(self) -> int:
        return len(self.captions_tokens)

    def __getitem__(self, item):
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[item]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix


    






"""

    ВАЛИДАЦИЯ

"""

def filter_ngrams(output_text):
    a_pos = output_text.find(' A:')
    sec_a_pos = output_text.find(' A:', a_pos + 1)
    return output_text[:sec_a_pos]

def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt='',
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.98,
        temperature=1.,
        stop_token = '<|endoftext|>',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if not tokens:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    #print('tokens',tokens)
                    tokens = tokens.unsqueeze(0).to(device)

            emb_tokens = model.gpt.transformer.wte(tokens)

            if embed is not None:
                generated = torch.cat((embed, emb_tokens), dim=1)
            else:
                generated = emb_tokens

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                #
                top_k = 2000
                top_p = 0.98
                #print(logits)
                #next_token = transformers.top_k_top_p_filtering(logits.to(torch.int64).unsqueeze(0), top_k=top_k, top_p=top_p)
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)

                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)

                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())

            output_text = tokenizer.decode(output_list)
            output_text = filter_ngrams(output_text)
            generated_list.append(output_text)

    return generated_list[0]

