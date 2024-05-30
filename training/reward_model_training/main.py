import argparse
import deepspeed
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
    Adafactor
)
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.accelerator import get_accelerator

from dschat.model.reward_model import RewardModel
from dschat.data.data_utils import create_dataset, DataCollatorReward
from dschat.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, \
    get_optimizer_grouped_parameters, save_zero_three_model
from dschat.ds_utils import get_train_ds_config


def parse_args():
    parser = argparse.ArgumentParser(description="Reward Model Training")
    parser.add_argument('--data_path', type=str, default='./data_input')
    parser.add_argument('--data_output_path', type=str, default='./data_output')
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument("--offload_ratio", type=float, default=0.3)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument('--offload', action='store_true')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16'])
    parser.add_argument('--zero_stage', type=int, default=0)

    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=0)
    parser.add_argument("--eval_iters", type=int, default=100, help="Maximum evaluation iterations")
    ## low precision
    parser.add_argument(
        '--compute_fp32_loss',
        action='store_true',
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

    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step2_tensorboard")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def save_checkpoint(args, model, tokenizer, epoch, reward_score, acc):
    reward_score = round(reward_score, 3)
    acc = round(acc, 3)
    dir_name = "epoch_" + str(epoch) + "_reward_score_" + str(reward_score) + "_accuracy_" + str(acc)
    cur_save_path = os.path.join(args.output_dir, dir_name)
    if args.global_rank == 0 and not os.path.exists(cur_save_path):
        os.makedirs(cur_save_path, exist_ok=True)
    print_rank_0('saving the final model ...', args.global_rank)
    if args.global_rank == 0:
        save_hf_format(model, tokenizer, args, sub_folder=dir_name)
    if args.zero_stage == 3:
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
                                    tb_name="step2_model")
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
    ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'

    train_phase = "reward"
    train_dataset, eval_dataset = create_dataset(args.data_path, args.data_output_path, train_phase, tokenizer,
                                                 args.max_seq_len, reload=False)
    # DataLoaders creation:
    data_collator = DataCollatorReward()
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    rm_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, low_cpu_mem_usage=True,
                                                    trust_remote_code=True).model
    rm_model = RewardModel(rm_model, tokenizer, compute_fp32_loss=args.compute_fp32_loss)

    def evaluation_reward(model, dataloader, eval_iters):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        chosen_scores = 0.
        rejected_scores = 0.
        for _step, _batch in enumerate(dataloader):
            _batch = to_device(_batch, device)
            with torch.no_grad():
                _outputs = model(**_batch)

            chosen = _outputs["chosen_mean_scores"]
            rejected = _outputs["rejected_mean_scores"]
            correct_predictions += (chosen > rejected).sum()
            total_predictions += chosen.shape[0]
            chosen_scores += _outputs["chosen_mean_scores"].mean().float()
            rejected_scores += _outputs["rejected_mean_scores"].mean().float()
            if (_step + 1) == eval_iters:
                break
        _acc = correct_predictions / total_predictions
        chosen_scores = chosen_scores / (_step + 1)
        rejected_scores = rejected_scores / (_step + 1)
        try:
            _acc = get_all_reduce_mean(_acc).item()
            chosen_scores = get_all_reduce_mean(chosen_scores).item()
            rejected_scores = get_all_reduce_mean(rejected_scores).item()
        except:
            pass
        model.train()
        return chosen_scores, rejected_scores, _acc

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(rm_model, args.weight_decay)

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
        num_training_steps=args.num_train_epochs * len(train_dataloader),
    )

    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    print_rank_0(
        f"***** Evaluating reward, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    reward_score, reject_score, acc = evaluation_reward(
        rm_model, eval_dataloader, args.eval_iters)
    print_rank_0(
        f"chosen_last_scores (higher is better) : {reward_score}, "
        f"rejected_last_scores (lower is better) : {reject_score}, "
        f"acc (higher is better) : {acc}", args.global_rank)
    torch.cuda.empty_cache()

    total_micro_steps = 0
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        rm_model.train()
        mean_loss = 0
        mean_chosen_scores = 0
        mean_rejected_scores = 0
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = rm_model(**batch, use_cache=False)
            loss = outputs["loss"]
            chosen_scores = outputs["chosen_mean_scores"]
            rejected_scores = outputs["rejected_mean_scores"]
            rm_model.backward(loss)
            rm_model.step()
            mean_loss += loss.item()
            mean_chosen_scores += torch.mean(chosen_scores).item()
            mean_rejected_scores += torch.mean(rejected_scores).item()
            total_micro_steps += 1

            total_steps = total_micro_steps // args.gradient_accumulation_steps
            if step % 10 == 0:
                print_rank_0(
                    f"Step {step} with loss {mean_loss / (step + 1)}, "
                    f"mean_chosen_scores {mean_chosen_scores / (step + 1)}, "
                    f"mean_rejected_scores {mean_rejected_scores / (step + 1)}",
                    args.global_rank)
            if args.eval_interval and (total_steps % args.eval_interval == 0):
                print_rank_0(f"Iter {total_steps}: Evaluating reward", args.global_rank)
                reward_score, reject_score, acc = evaluation_reward(rm_model, eval_dataloader, args.eval_iters)
                print_rank_0(
                    f"Iter {total_steps}: c_scores: {reward_score}, r_scores: {reject_score}, "
                    f"diff: {reward_score - reject_score}, acc: {acc}",
                    args.global_rank)

        print_rank_0(
            f"***** Evaluating reward, Epoch {epoch + 1}/{args.num_train_epochs} *****",
            args.global_rank)
        reward_score, reject_score, acc = evaluation_reward(
            rm_model, eval_dataloader, args.eval_iters)
        print_rank_0(
            f"chosen_last_scores (higher is better) : {reward_score}, "
            f"rejected_last_scores (lower is better) : {reject_score}, "
            f"acc (higher is better) : {acc}", args.global_rank)
        save_checkpoint(args, rm_model, tokenizer, epoch, reward_score, acc)


if __name__ == "__main__":
    main()
