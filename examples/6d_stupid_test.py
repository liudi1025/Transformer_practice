from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import defaultdict
import time

#start_time = time.time()

orig_text = ['I like bananas.', 'Yesterday the mailman came by!', 'Do you enjoy cookies?']*10
edit_text = ['Do you?', 'He delivered a mystery package.', 'My grandma just baked some!']

tokenizer = AutoTokenizer.from_pretrained("/data1/liudi/Transformers/examples/output/desc")
model = AutoModelForSequenceClassification.from_pretrained("/data1/liudi/Transformers/examples/output/desc")
start_time = time.time()
desc = tokenizer.batch_encode_plus(orig_text, return_tensors="pt")
desc_classification_logits = model(**desc)[0]
results = torch.softmax(desc_classification_logits, dim=1).tolist()
#print(results)
end_time = time.time()
print(end_time-start_time)



def encode_batch_plus(batch,
                      batch_pair=None,
                      pad_to_batch_length=False,
                      return_tensors=None,
                      return_token_type_ids=True,
                      return_attention_mask=True,
                      return_special_tokens_mask=False,
                      **kwargs):
    if pad_to_batch_length and 'pad_to_max_length' in kwargs and kwargs['pad_to_max_length']:
        raise ValueError("'pad_to_batch_length' and 'pad_to_max_length' cannot be used simultaneously.")

    def merge_dicts(list_of_ds):
        d = defaultdict(list)
        for _d in list_of_ds:
            for _k, _v in _d.items():
                d[_k].append(_v)

        return dict(d)

    # gather all encoded inputs in a list of dicts
    encoded = []
    batch_pair = [None] * len(batch) if batch_pair is None else batch_pair
    for firs_sent, second_sent in zip(batch, batch_pair):
        # return_tensors=None: don't convert to tensors yet. Do that manually as the last step
        encoded.append(TOKENIZER.encode_plus(firs_sent,
                                             second_sent,
                                             return_tensors=None,
                                             return_token_type_ids=return_token_type_ids,
                                             return_attention_mask=return_attention_mask,
                                             return_special_tokens_mask=return_special_tokens_mask,
                                             **kwargs))

    # convert list of dicts in a single merged dict
    encoded = merge_dicts(encoded)

    if pad_to_batch_length:
        max_batch_len = max([len(l) for l in encoded['input_ids']])

        if TOKENIZER.padding_side == 'right':
            if return_attention_mask:
                encoded['attention_mask'] = [mask + [0] * (max_batch_len - len(mask)) for mask in
                                             encoded['attention_mask']]
            if return_token_type_ids:
                encoded["token_type_ids"] = [ttis + [TOKENIZER.pad_token_type_id] * (max_batch_len - len(ttis)) for ttis
                                             in encoded['token_type_ids']]
            if return_special_tokens_mask:
                encoded['special_tokens_mask'] = [stm + [1] * (max_batch_len - len(stm)) for stm in
                                                  encoded['special_tokens_mask']]
            encoded['input_ids'] = [ii + [TOKENIZER.pad_token_id] * (max_batch_len - len(ii)) for ii in
                                    encoded['input_ids']]
        elif TOKENIZER.padding_side == 'left':
            if return_attention_mask:
                encoded['attention_mask'] = [[0] * (max_batch_len - len(mask)) + mask for mask in
                                             encoded['attention_mask']]
            if return_token_type_ids:
                encoded['token_type_ids'] = [[TOKENIZER.pad_token_type_id] * (max_batch_len - len(ttis)) for ttis in
                                             encoded['token_type_ids']]
            if return_special_tokens_mask:
                encoded['special_tokens_mask'] = [[1] * (max_batch_len - len(stm)) + stm for stm in
                                                  encoded['special_tokens_mask']]
            encoded['input_ids'] = [[TOKENIZER.pad_token_id] * (max_batch_len - len(ii)) + ii for ii in
                                    encoded['input_ids']]
        else:
            raise ValueError(f"Invalid padding strategy: {TOKENIZER.padding_side}")

    if return_tensors is not None:
        if return_tensors in {'pt', 'tf'}:
            encoded['input_ids'] = tf.constant(encoded['input_ids']) if return_tensors == 'tf' \
                else torch.tensor(encoded['input_ids'])
            if 'attention_mask' in encoded:
                encoded['attention_mask'] = tf.constant(encoded['attention_mask']) if return_tensors == 'tf' \
                    else torch.tensor(encoded['attention_mask'])
            if 'token_type_ids' in encoded:
                encoded['token_type_ids'] = tf.constant(encoded['token_type_ids']) if return_tensors == 'tf' \
                    else torch.tensor(encoded['token_type_ids'])
            if 'special_tokens_mask' in encoded:
                encoded['special_tokens_mask'] = tf.constant(encoded['special_tokens_mask']) if return_tensors == 'tf' \
                    else torch.tensor(encoded['special_tokens_mask'])
            # should num_truncated_tokens, overflowing_tokens also be converted to tensors?
            # if yes then this could be generalised in a for loop/dict comprehension converting all k,v to k,tensor(v)
        else:
            raise ValueError(f"Cannot return tensors with value '{return_tensors}'")

    return encoded
