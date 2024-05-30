import argparse
import os
import sys
import torch
import deepspeed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    Adafactor
)
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator
from dschat.data.data_utils import create_dataset
from dschat.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, \
    get_optimizer_grouped_parameters, save_zero_three_model
from dschat.ds_utils import get_train_ds_config
from dschat.model.model_utils import causal_lm_model_to_fp32_loss
from dschat.perf import print_throughput


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--do_eval', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='./data_input')
    parser.add_argument('--data_output_path', type=str, default='./data_output')
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--num_train_steps", type=int, default=500000)
    parser.add_argument("--checkpoint_steps", type=int, default=10000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument("--offload_ratio", type=float, default=0.3)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--offload', action='store_true')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['fp16', 'bf16', 'fp32'])
    parser.add_argument('--zero_stage', type=int, default=0)
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard', action='store_true')
    parser.add_argument('--tensorboard_path', type=str)
    ## low precision
    parser.add_argument('--compute_fp32_loss', action='store_true',
                        help='Relevant for low precision dtypes (fp16, bf16, etc.). '
                             'If specified, loss is calculated in fp32.')
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def save_checkpoint(args, model, tokenizer, epoch, step, ppl):
    ppl = round(ppl, 4)
    cur_save_path = os.path.join(args.output_dir, "epoch_" + str(epoch) + "_step_" + str(step) + "_ppl_" + str(ppl))
    if args.global_rank == 0 and not os.path.exists(cur_save_path):
        os.makedirs(cur_save_path, exist_ok=True)
    print_rank_0('saving the final model ...', args.global_rank)
    if args.global_rank == 0:
        save_hf_format(model, tokenizer, args,
                       sub_folder="epoch_" + str(epoch) + "_step_" + str(step) + "_ppl_" + str(ppl))
    if args.zero_stage == 3:
        # For zero stage 3, each gpu only has a part of the model, so we need a special save function
        save_zero_three_model(model,
                              args.global_rank,
                              cur_save_path,
                              zero_stage=args.zero_stage)


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    dtype=args.dtype,
                                    offload_ratio=args.offload_ratio,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step1_model")
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
    ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    train_phase = "sft"
    # Prepare the data
    train_dataset, eval_dataset = create_dataset(
        args.data_path,
        args.data_output_path,
        train_phase,
        tokenizer,
        args.max_seq_len,
        reload=False)

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True, trust_remote_code=True)

    if args.compute_fp32_loss:
        print_rank_0(
            f"Using model {model.__class__.__name__} with loss in fp32",
            args.global_rank)
        causal_lm_model_to_fp32_loss(model)

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            losses = get_all_reduce_mean(losses)
        except:
            pass
        try:
            perplexity = torch.exp(losses).item()
        except OverflowError:
            perplexity = float("inf")
        model.train()
        return perplexity, losses.item()

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args.weight_decay)

    optimizer = None
    if args.optimizer == "adam":
        optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))
    elif args.optimizer == "adam-cpu":
        optimizer = DeepSpeedCPUAdam(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))
    elif args.optimizer == "adafactor":
        optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, scale_parameter=False,
                              relative_step=False)
    else:
        print("no optimizer found!")

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_steps,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    if args.do_eval:
        print_rank_0(
            f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
            args.global_rank)
        perplexity, eval_loss = evaluation(model, eval_dataloader)
        print_rank_0(f"ppl: {perplexity}, loss: {eval_loss}", args.global_rank)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        import time
        for step, batch in enumerate(train_dataloader):
            start = time.time()
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            ppl = round(torch.exp(loss).item(), 4)
            print_rank_0(f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, ppl = {ppl}")
            model.backward(loss)
            model.step()
            end = time.time()
            if torch.distributed.get_rank() == 0:
                print_throughput(model.model, args, end - start, args.global_rank)
            if step and step % args.checkpoint_steps == 0:
                perplexity = -1
                if args.do_eval:
                    print_rank_0(f"***** Evaluating perplexity *****", args.global_rank)
                    perplexity, eval_loss = evaluation(model, eval_dataloader)
                    print_rank_0(f"ppl: {perplexity}, loss: {eval_loss}", args.global_rank)
                save_checkpoint(args, model, tokenizer, epoch, step, perplexity)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
