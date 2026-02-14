import argparse
import os
import shutil
import sys
import time
from functools import partial

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

import deepspeed
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.naidan import SIDAForCausalLM
from model.llava import conversation as conversation_lib
from utils.SID_Set import collate_fn, CustomDataset
from utils.batch_sampler import BatchSampler
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    AverageMeter,
    ProgressMeter,
    Summary,
    dict_to_cuda,
    intersectionAndUnionGPU,
)

import random
import torch.distributed as dist


def parse_args(args):
    parser = argparse.ArgumentParser(description="SIDA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")

    parser.add_argument(
        "--version",
        default="/project/shih/nz85/SIDA-main/ck/SIDA-7B",
    )

    parser.add_argument("--vis_save_path", default="./vis_output", type=str)

    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)

    parser.add_argument("--lora_r", default=0, type=int)
    # parser.add_argument("--lora_r", default=8, type=int)

    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--val_dataset", default="val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="sida", type=str)

    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=2, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.00001, type=float)

    # Stage-specific arguments
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes for classification")
    parser.add_argument("--use_stage1_cls", action="store_true", default=True)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=1.0, type=float)
    parser.add_argument("--bce_loss_weight", default=1.0, type=float)
    parser.add_argument("--cls_loss_weight", default=1.0, type=float)
    parser.add_argument("--mask_loss_weight", default=1.0, type=float)

    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)

    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)

    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)

    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)

    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)

    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)

    # Early stopping (fixed + extended)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=10,
        help="Stop if no improvement for this many validation runs (0 = disabled).",
    )
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="combined",
        choices=["giou", "f1", "combined"],
        help="Which score to monitor for early stopping.",
    )
    parser.add_argument(
        "--early_stop_alpha",
        type=float,
        default=0.7,
        help="Weight for gIoU in combined early stop score: alpha*giou + (1-alpha)*macro_f1.",
    )

    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )

    return parser.parse_args(args)


def compute_main_score(args, giou: float, macro_f1: float) -> float:
    if args.early_stop_metric == "giou":
        return float(giou)
    if args.early_stop_metric == "f1":
        return float(macro_f1)
    # combined
    return float(args.early_stop_alpha * giou + (1.0 - args.early_stop_alpha) * macro_f1)


def main(cli_args):
    args = parse_args(cli_args)

    deepspeed.init_distributed()

    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[CLS]")
    tokenizer.add_tokens("[SEG]")

    args.cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "cls_loss_weight": args.cls_loss_weight,
        "mask_loss_weight": args.mask_loss_weight,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "cls_token_idx": args.cls_token_idx,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    model = SIDAForCausalLM.from_pretrained(
        args.version,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        **model_args,
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    print("\nChecking specific components:")
    for component in ["cls_head", "sida_fc1", "attention_layer", "text_hidden_fcs"]:
        matching_params = [n for n, _ in model.named_parameters() if component in n]
        if matching_params:
            print(f"Found {component} in parameters: {matching_params}")
        else:
            print(f"Component not found: {component}")

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    if not args.eval_only:
        model.get_model().initialize_sida_modules(model.get_model().config)

    # Freeze vision and projector
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    lora_r = args.lora_r
    base = model.get_model()

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze GAG
    if hasattr(base, "gag") and base.gag is not None:
        for p in base.gag.parameters():
            p.requires_grad = True
        base.gag.train()
        print("✅ Unfroze: base.gag")
    else:
        print("⚠️ base.gag not found")

    # Find decoder module
    decoder_module = None
    decoder_candidates = []
    for name, module in base.named_modules():
        ln = name.lower()
        if "mask_decoder" in ln or (ln.endswith("decoder") and "decoder" in ln):
            decoder_candidates.append(name)

    print("Decoder candidates:")
    for n in decoder_candidates[:50]:
        print("  ", n)

    for name, module in base.named_modules():
        if "mask_decoder" in name.lower():
            decoder_module = module
            print("✅ Using decoder:", name)
            break

    if decoder_module is None:
        for name, module in base.named_modules():
            if name.lower().endswith("decoder"):
                decoder_module = module
                print("✅ Using decoder:", name)
                break

    if decoder_module is None:
        raise RuntimeError("❌ Could not find decoder. Check printed 'Decoder candidates' above.")

    for p in decoder_module.parameters():
        p.requires_grad = True
    decoder_module.train()
    print("✅ Unfroze: decoder module")

    # LoRA (optional)
    if lora_r > 0:

        def find_linear_layers(m, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in m.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        x not in name
                        for x in [
                            "visual_model",
                            "vision_tower",
                            "mm_projector",
                            "text_hidden_fcs",
                            "cls_head",
                            "sida_fc1",
                            "attention_layer",
                        ]
                    )
                    and any(x in name for x in lora_target_modules)
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_target_modules = find_linear_layers(model, args.lora_target_modules.split(","))
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # Make sure lm_head + some text parts remain frozen
    for n, p in model.named_parameters():
        if "lm_head" in n:
            p.requires_grad = False
        if any(x in n for x in ["embed_tokens", "text_hidden_fcs", "cls_head", "sida_fc1", "attention_layer"]):
            p.requires_grad = False

    print("Checking trainable parameters:")
    total_params = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"Trainable: {n} with {p.numel()} parameters")
            total_params += p.numel()
    print(f"Total trainable parameters: {total_params}")

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    train_dataset = CustomDataset(
        base_image_dir=args.dataset_dir,
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,
        split="train",
        precision=args.precision,
        image_size=args.image_size,
        per_class_limit=35000,
        seed=0,
    )
    print(f"\nInitializing datasets:")
    print(f"Training split size: {len(train_dataset)}")

    if not args.no_eval:
        val_dataset = CustomDataset(
            base_image_dir=args.dataset_dir,
            tokenizer=tokenizer,
            vision_tower=args.vision_tower,
            split="validation",
            precision=args.precision,
            image_size=args.image_size,
        )
        print(f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples.")
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    print("----------------trainable params----------------")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print("train", n, p.shape)
    print("----------------end----------------")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
            "loss_scale": 0,
            "initial_scale_power": 12,
            "loss_scale_window": 1000,
            "min_loss_scale": 1,
            "hysteresis": 2,
        },
        "bf16": {"enabled": args.precision == "bf16"},
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
        "zero_allow_untested_optimizer": True,
    }

    batch_sampler = BatchSampler(
        dataset=train_dataset,
        batch_size=ds_config["train_micro_batch_size_per_gpu"],
        world_size=torch.cuda.device_count(),
        rank=args.local_rank,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
            cls_token_idx=args.cls_token_idx,
        ),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    basic_optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=0.0,
        betas=(args.beta1, args.beta2),
    )

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=basic_optimizer,
        config=ds_config,
        training_data=None,
    )

    # Auto resume
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        print(f"resume training from {args.resume}, start from epoch {args.start_epoch}")

    # Validation loader
    if val_dataset is not None:
        val_sampler = BatchSampler(
            dataset=val_dataset,
            batch_size=args.val_batch_size,
            world_size=torch.cuda.device_count(),
            rank=args.local_rank,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )
    else:
        val_loader = None

    train_iter = iter(train_loader)

    # Best tracking (fixed)
    best_acc = 0.0
    best_giou = 0.0
    best_main_score = -1e9
    best_ciou = 0.0
    epochs_without_improvement = 0

    if args.eval_only:
        if val_loader is None:
            raise RuntimeError("eval_only=True but no val_loader (no_eval=True).")
        acc, giou, ciou, per_class_metrics, macro_f1 = validate(val_loader, model_engine, 0, writer, args)
        return

    validation_epochs = [1, 3, 5, 7, 10]
    if args.local_rank == 0:
        print(f"\nTraining Configuration:")
        print(f"Total epochs: {args.epochs}")
        print(f"Validation will be performed after epochs: {validation_epochs}")
        print(f"Early stop metric: {args.early_stop_metric}, patience: {args.early_stop_patience}")

    for epoch in range(args.start_epoch, args.epochs):
        train_iter = train(train_loader, model_engine, epoch, scheduler, writer, train_iter, args)

        if (epoch + 1) in validation_epochs:
            if args.local_rank == 0:
                print(f"\nPerforming validation after epoch {epoch + 1}")

            if not args.no_eval:
                acc, giou, ciou, per_class_metrics, macro_f1 = validate(val_loader, model_engine, epoch, writer, args)

                main_score = compute_main_score(args, giou=giou, macro_f1=macro_f1)

                # "best" flags
                is_best_acc = acc >= best_acc
                is_best_iou = giou >= best_giou
                is_best_main = main_score >= best_main_score

                if is_best_acc:
                    best_acc = acc
                if is_best_iou:
                    best_giou = giou
                    best_ciou = ciou
                if is_best_main:
                    best_main_score = main_score

                # Early stopping uses ONLY main_score now (fixed)
                if args.early_stop_patience > 0:
                    if not is_best_main:
                        epochs_without_improvement += 1
                        if args.local_rank == 0:
                            print(
                                f"Early stopping: no improvement for "
                                f"{epochs_without_improvement}/{args.early_stop_patience} "
                                f"(metric={args.early_stop_metric}, main_score={main_score:.6f}, best={best_main_score:.6f})"
                            )
                        if epochs_without_improvement >= args.early_stop_patience:
                            if args.local_rank == 0:
                                print(f"Early stopping triggered after epoch {epoch + 1}. Exiting.")
                            break
                    else:
                        epochs_without_improvement = 0

                if args.local_rank == 0:
                    print(f"Current accuracy: {acc:.4f}%, Best accuracy: {best_acc:.4f}%")
                    print(f"Current gIoU: {giou:.4f}, Best gIoU: {best_giou:.4f}")
                    print(f"Current macro-F1: {macro_f1:.4f}, Best main_score: {best_main_score:.6f}")

                # Save checkpoint if best by main score (recommended)
                if args.no_eval or is_best_main:
                    save_dir = os.path.join(args.log_dir, "ckpt_model")
                    if args.local_rank == 0:
                        torch.save(
                            {"epoch": epoch, "best_main_score": best_main_score, "best_acc": best_acc, "best_giou": best_giou},
                            os.path.join(args.log_dir, f"meta_log_best_main_{best_main_score:.6f}.pth"),
                        )
                        if os.path.exists(save_dir):
                            shutil.rmtree(save_dir)
                    torch.distributed.barrier()
                    model_engine.save_checkpoint(save_dir)
            else:
                if args.local_rank == 0:
                    print("no_eval=True, skipping validation metrics and early stopping.")
        else:
            if args.local_rank == 0:
                print(f"Epoch {epoch + 1} completed. Skipping validation.")

        # Save final epoch
        if epoch == args.epochs - 1:
            save_dir = os.path.join(args.log_dir, "final_checkpoint")
            if args.local_rank == 0:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)
            if args.local_rank == 0:
                print(f"\nTraining completed. Final checkpoint saved to {save_dir}")


def train(train_loader, model, epoch, scheduler, writer, train_iter, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    cls_losses = AverageMeter("ClsLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [batch_time, losses, cls_losses, mask_bce_losses, mask_dice_losses, mask_losses],
        prefix=f"Epoch: [{epoch}]",
    )

    model.train()
    end = time.time()

    for global_step in range(args.steps_per_epoch):
        model.zero_grad()

        for _ in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)

            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            cls_loss = output_dict["cls_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            cls_losses.update(cls_loss.item(), input_dict["images"].size(0))

            if input_dict["cls_labels"][0] == 2:
                mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
                mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
                mask_losses.update(mask_loss.item(), input_dict["images"].size(0))

            model.backward(loss)
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                cls_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if args.local_rank == 0 and writer is not None:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/cls_loss", cls_losses.avg, global_step)
                writer.add_scalar("train/mask_bce_loss", mask_bce_losses.avg, global_step)
                writer.add_scalar("train/mask_dice_loss", mask_dice_losses.avg, global_step)
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, global_step)
                writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, global_step)

            batch_time.reset()
            data_time.reset()
            losses.reset()
            cls_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0 and writer is not None:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def validate(val_loader, model_engine, epoch, writer, args, sample_ratio=None):
    model_engine.eval()

    correct = 0
    total = 0
    num_classes = 3
    confusion_matrix = torch.zeros(num_classes, num_classes, device="cuda")

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    total_batches = len(val_loader)
    sample_indices = None
    if sample_ratio is not None:
        num_batches = max(1, int(total_batches * sample_ratio))
        sample_indices = set(random.sample(range(total_batches), num_batches))
        print(f"\nValidating on {num_batches}/{total_batches} randomly sampled batches...")

    for batch_idx, input_dict in enumerate(tqdm.tqdm(val_loader)):
        if sample_indices is not None and batch_idx not in sample_indices:
            continue

        if batch_idx == 0:
            print("\nFirst validation batch details:")
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key} shape: {tuple(value.shape)}")
                elif isinstance(value, list):
                    print(f"{key} length: {len(value)}")

        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict)

        if total == 0:
            print("\nProcessing first batch:")
            print("Input dict keys:", list(input_dict.keys()))

        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        input_dict["inference"] = True
        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        logits = output_dict["logits"]
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        cls_labels = input_dict["cls_labels"]

        correct += (preds == cls_labels).sum().item()
        total += cls_labels.size(0)

        for t, p in zip(cls_labels, preds):
            confusion_matrix[t.long(), p.long()] += 1

        # Segmentation validation only when cls_label == 2
        if cls_labels[0] == 2:
            pred_masks = output_dict["pred_masks"]
            masks_list = output_dict["gt_masks"][0].int()
            output_list = (pred_masks[0] > 0).int()
            assert len(pred_masks) == 1

            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i, output_i in zip(masks_list, output_list):
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(),
                    mask_i.contiguous(),
                    2,
                    ignore_index=255,
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0  # no-object target

            intersection = intersection.cpu().numpy()
            union = union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]

            intersection_meter.update(intersection)
            union_meter.update(union)
            acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    # Reduce metrics
    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = float(iou_class[1]) if len(iou_class) > 1 else 0.0
    giou = float(acc_iou_meter.avg[1]) if len(acc_iou_meter.avg) > 1 else 0.0

    accuracy = (correct / max(total, 1)) * 100.0

    # Classification per-class metrics
    confusion_matrix = confusion_matrix.cpu()
    class_names = ["Real", "Full Synthetic", "Tampered"]
    per_class_metrics = {}

    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp

        total_class_samples = confusion_matrix[i, :].sum()

        class_accuracy = float(tp / total_class_samples) if total_class_samples > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class_metrics[class_names[i]] = {
            "accuracy": class_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro_f1 = float(np.mean([m["f1"] for m in per_class_metrics.values()]))

    # Pixel accuracy / additional metrics (your original)
    pixel_correct = float(intersection_meter.sum[1]) if len(intersection_meter.sum) > 1 else 0.0
    pixel_total = float(union_meter.sum[1]) if len(union_meter.sum) > 1 else 0.0
    pixel_accuracy = (pixel_correct / (pixel_total + 1e-10)) * 100.0 if pixel_total > 0 else 0.0

    iou = ciou
    # NOTE: this "f1_score" is mixing cls-accuracy with mask-iou; keep if you want, but it's not macro-F1.
    f1_score = (
        2 * (iou * (accuracy / 100.0)) / (iou + (accuracy / 100.0) + 1e-10)
        if (iou + (accuracy / 100.0)) > 0
        else 0.0
    )

    avg_precision = float(np.mean([m["precision"] for m in per_class_metrics.values()]))
    avg_recall = float(np.mean([m["recall"] for m in per_class_metrics.values()]))
    auc_approx = avg_precision * avg_recall

    # Log metrics
    if args.local_rank == 0 and writer is not None:
        writer.add_scalar("val/accuracy", accuracy, epoch)
        writer.add_scalar("val/macro_f1", macro_f1, epoch)

        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)

        writer.add_scalar("val/pixel_accuracy", pixel_accuracy, epoch)
        writer.add_scalar("val/iou", iou, epoch)
        writer.add_scalar("val/f1_score", f1_score, epoch)
        writer.add_scalar("val/auc_approx", auc_approx, epoch)

        # log per-class
        for class_name, metrics in per_class_metrics.items():
            safe_name = class_name.lower().replace("/", "_").replace(" ", "_")
            for metric_name, value in metrics.items():
                writer.add_scalar(f"val/{safe_name}_{metric_name}", value, epoch)

        # log main score used for early stopping
        main_score = compute_main_score(args, giou=giou, macro_f1=macro_f1)
        writer.add_scalar("val/early_stop_main_score", main_score, epoch)

        validation_type = "Full" if sample_ratio is None else f"Sampled ({sample_ratio*100:.1f}%)"
        print(f"\n{validation_type} Validation Results:")
        print(f"gIoU: {giou:.4f}, cIoU: {ciou:.4f}")
        print(f"Classification Accuracy: {accuracy:.4f}%")
        print(f"Macro-F1 (classification): {macro_f1:.4f}")
        print(f"Pixel Accuracy: {pixel_accuracy:.4f}%")
        print(f"IoU: {iou:.4f}")
        print(f"F1 Score (mixed): {f1_score:.4f}")
        print(f"Approximate AUC: {auc_approx:.4f}")
        print(f"Total correct classifications: {correct}")
        print(f"Total classification samples: {total}")

        print("\nPer-Class Metrics:")
        for class_name, metrics in per_class_metrics.items():
            print(f"\n{class_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")

        print("\nConfusion Matrix:")
        print(f"{'':20}", end="")
        for name in class_names:
            print(f"{name:>12}", end="")
        print()
        for i, class_name in enumerate(class_names):
            print(f"{class_name:20}", end="")
            for j in range(num_classes):
                print(f"{confusion_matrix[i, j]:12.0f}", end="")
            print()

    return accuracy, giou, ciou, per_class_metrics, macro_f1


if __name__ == "__main__":
    main(sys.argv[1:])
