import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup

from gpt2_pre_data import load_doc_data
from text_llm.DocContent import DocContent

GENERATED_LENGTH = 30
MAX_DATA_LOADED = 2


# Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None


def training_pipeline(
        data_loader,
        t_model,
        tokenizer,
        batch_size=16,
        epochs=20,
        lr=2e-5,
        max_seq_len=400,
        warmup_steps=200,
        output_dir=".",
        output_prefix="wreckgar",
        save_model_on_epoch=False,
):
    acc_steps = 100
    device = torch.device("cuda")
    t_model = t_model.cuda()
    t_model.train()

    optimizer = AdamW(t_model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(data_loader, batch_size=1, shuffle=True)
    loss = 0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):
        print(f"Training epoch {epoch}")
        print(loss)
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = t_model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                t_model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                t_model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )

    return t_model


# ### Actual Training
def get_gen_model(df, gpt2_type='gpt2', model_name='model/model_v1.pt'):
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)

    # Train the model on the specific data we have
    model = GPT2LMHeadModel.from_pretrained(gpt2_type)

    # training
    t_model = training_pipeline(DocContent(df, tokenizer), model, tokenizer)

    # Save the model to a pkl or something it can be reused later on
    torch.save(t_model, model_name)

    return t_model, tokenizer


def gen_train():
    df, _ = load_doc_data(max_data_loaded=MAX_DATA_LOADED)
    get_gen_model(df=df)


if __name__ == '__main__':
    gen_train()
