from datetime import datetime

import numpy as np
import torch

from latex_decompiler.loss.motif_loss import motif_loss_types
from latex_decompiler.splicing.utils import topk_both

from .dataset import DATA_TYPE_MAP
from .model import full_model_types, motif_model_types
from sparling import SparsityUpdateOptimizer, suo_types
from .utils import (
    construct,
    load_model,
    run_batched_fn,
    save_model,
    strip_start_and_end_tokens,
)


def train_generic(
    *,
    path,
    get_model,
    get_optimizer,
    data_train,
    data_val,
    train_fn,
    val_fn,
    run_val,
    total_steps,
    batch_size,
    print_every,
    val_every,
    save_every,
    val_percent,
    done_fn=lambda model: False,
    device="cuda",
):
    step, optimizer_model = load_model(path)
    if step is None:
        step, model = 0, get_model().to(device)
        optimizer = get_optimizer(model)
        last_steps = None
    else:
        optimizer, model = optimizer_model["optimizer"], optimizer_model["model"]
        last_steps = optimizer_model.get("last_steps", None)
    if last_steps is None:
        last_steps = dict(save=step, val=step, print=step)
    losses = {}
    print("starting at step", step)
    while step < total_steps:
        xs, ys = zip(*[data_train[i] for i in range(step, step + batch_size)])
        step += batch_size
        optimizer.zero_grad()
        loss = train_fn(model, xs, ys)
        loss.backward()
        optimizer.step()
        losses[step] = loss.item()
        if step - last_steps["print"] >= print_every:
            last_steps["print"] = step
            print(
                f"[{datetime.now()}] {step}/{total_steps}: {np.mean(list(losses.values()))}"
            )
            save_model(dict(losses=losses), path, step, key="losses")
            losses = {}
        if step - last_steps["val"] >= val_every:
            last_steps["val"] = step

            model = model.eval()

            all_ys, all_ys_pred = run_batched_fn(
                lambda xs: run_val(model, xs),
                data_val,
                max(1, val_percent * val_every),
                batch_size,
            )

            val_fn_result = val_fn(
                model=model,
                optimizer=optimizer,
                all_ys=all_ys,
                all_ys_pred=all_ys_pred,
                step=step,
            )

            model = model.train()
            save_model(
                dict(
                    all_ys=all_ys, all_ys_pred=all_ys_pred, val_fn_result=val_fn_result
                ),
                path,
                step,
                key="val_result",
            )

        if step - last_steps["save"] >= save_every:
            last_steps["save"] = step
            save_model(
                dict(optimizer=optimizer, model=model, last_steps=last_steps),
                path,
                step,
            )

        if done_fn(model):
            break

    return model


def train_latex_e2e(
    *,
    path,
    architecture,
    data_spec,
    train_seed,
    val_seed,
    total_steps,
    batch_size,
    print_every,
    val_every,
    val_callback=lambda *args, **kwargs: {},
    val_percent=0.1,
    max_length=30,
    lr=1e-4,
    suo_spec=dict(type="NoopSUO"),
    motif_loss_specs=[],
    done_at_density=-float("inf"),
    validation_spec=dict(type="TokenValidation"),
    device="cuda",
):
    validation = construct(
        dict(TokenValidation=TokenValidation, TopKValidation=TopKValidation),
        validation_spec,
    )
    motif_losses = [construct(motif_loss_types(), spec) for spec in motif_loss_specs]

    def train_fn(model, xs, ys):
        xs = torch.tensor(xs).float().to(device)
        return model.forward_train(xs, ys, motif_losses=motif_losses)

    def run_val(model, xs):
        return model.forward_test(torch.tensor(xs).float().to(device), max_length)

    def val_fn(*, model, optimizer, all_ys, all_ys_pred, step):
        acc, all_ys_pred = validation.compute_accuracy(all_ys, all_ys_pred)
        print(f"Accuracy: {acc:.2%}")

        if isinstance(optimizer, SparsityUpdateOptimizer):
            optimizer.update_sparsity(model, step, acc_info=dict(acc=acc))

        validation.report(all_ys, all_ys_pred)

        return dict(
            accuracy=acc,
            **val_callback(
                model=model, acc=acc, all_ys=all_ys, all_ys_pred=all_ys_pred
            ),
        )

    def done_fn(model):
        density = 1 - model.sparsity_value
        return density < done_at_density

    return train_generic(
        path=path,
        get_model=lambda: construct(full_model_types(), architecture),
        get_optimizer=lambda model: construct(
            suo_types(), suo_spec, optimizer=torch.optim.Adam(model.parameters(), lr=lr)
        ),
        data_train=construct(DATA_TYPE_MAP, data_spec, seed=train_seed),
        data_val=construct(DATA_TYPE_MAP, data_spec, seed=val_seed),
        train_fn=train_fn,
        val_fn=val_fn,
        run_val=run_val,
        total_steps=total_steps,
        batch_size=batch_size,
        print_every=print_every,
        save_every=val_every,
        val_every=val_every,
        val_percent=val_percent,
        done_fn=done_fn,
        device=device,
    )


def train_single_digit_motifs(
    *,
    path,
    architecture,
    data_spec,
    train_seed,
    val_seed,
    total_steps,
    batch_size,
    print_every,
    val_every,
    val_callback=lambda *args, **kwargs: {},
    val_percent=0.1,
    lr=1e-4,
):
    def reduce(yps):
        while len(yps.shape) > 2:
            yps = yps.max(dim=-1)[0]
        return yps

    def train_fn(model, xs, ys):
        xs = torch.tensor(xs).float().cuda()
        yps = model.run_motifs_without_post_sparse(xs)
        yps = reduce(yps)
        return torch.nn.CrossEntropyLoss()(yps, torch.tensor(ys).cuda())

    def run_val(model, xs):
        yps = model.run_motifs_without_post_sparse(torch.tensor(np.array(xs)).cuda())
        yps = reduce(yps)
        return yps.argmax(-1).cpu().numpy()

    def val_fn(*, model, optimizer, all_ys, all_ys_pred, step):
        acc = np.mean([y == yp for y, yp in zip(all_ys, all_ys_pred)])
        print(f"Accuracy: {acc:.2%}")

        def print_toks(tag, x):
            print(tag, x)

        print_toks("Actual   :", all_ys[:10])
        print_toks("Predicted:", all_ys_pred[:10])
        return dict(
            accuracy=acc,
            **val_callback(
                model=model, acc=acc, all_ys=all_ys, all_ys_pred=all_ys_pred
            ),
        )

    def done_fn(model):
        return False

    train_generic(
        path=path,
        get_model=lambda: construct(full_model_types(), architecture),
        get_optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=lr),
        data_train=construct(DATA_TYPE_MAP, data_spec, seed=train_seed),
        data_val=construct(DATA_TYPE_MAP, data_spec, seed=val_seed),
        train_fn=train_fn,
        val_fn=val_fn,
        run_val=run_val,
        total_steps=total_steps,
        batch_size=batch_size,
        print_every=print_every,
        save_every=val_every,
        val_every=val_every,
        val_percent=val_percent,
        done_fn=done_fn,
    )


def train_fixed_motifs(
    *,
    path,
    architecture,
    data_spec,
    train_seed,
    val_seed,
    total_steps,
    batch_size,
    print_every,
    val_every,
    val_callback=lambda *args, **kwargs: {},
    val_percent=0.1,
    lr=1e-4,
):
    def train_fn(model, xs, ys):
        xs = torch.tensor(xs).float().cuda()
        yps = model.run_motifs_without_post_sparse(xs)
        yps = torch.nn.functional.relu(yps)
        on_target = 0
        for batch_idx, stamps_each in enumerate(ys):
            for stamp in stamps_each:
                xmin, ymin, xmax, ymax = stamp["box"]
                on_target += yps[batch_idx, :, ymin : ymax + 1, xmin : xmax + 1].max()
        off_target = yps.sum() - on_target
        return off_target - on_target

    def run_val(model, xs):
        yps = model.run_motifs_without_post_sparse(
            torch.tensor(np.array(xs)).float().cuda()
        )
        yps = torch.nn.functional.relu(yps)
        return yps.cpu().numpy()

    def val_fn(*, model, optimizer, all_ys, all_ys_pred, step):
        print("val_fn")
        accs = []
        for y, yp in zip(all_ys, all_ys_pred):
            count = 0
            targets = {(target["symbol_id"], target["box"]) for target in y}
            flat_top = np.argpartition(yp.flatten(), -len(y))[-len(y) :]
            flat_top = np.unravel_index(flat_top, yp.shape)
            for motif_idx, i, j in zip(*flat_top):
                matches = [
                    target
                    for (motif_id, (xmin, ymin, xmax, ymax)), target in zip(
                        targets, targets
                    )
                    if motif_id == motif_idx and xmin <= j <= xmax and ymin <= i <= ymax
                ]
                if matches:
                    count += 1
                    targets -= {matches[0]}
            accs += [count / len(y)]
        acc = np.mean(accs)
        print(f"Accuracy: {acc:.2%}")
        return dict(
            accuracy=acc,
            **val_callback(model=model, acc=acc),
        )

    def done_fn(model):
        return False

    train_generic(
        path=path,
        get_model=lambda: construct(full_model_types(), architecture),
        get_optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=lr),
        data_train=construct(DATA_TYPE_MAP, data_spec, seed=train_seed),
        data_val=construct(DATA_TYPE_MAP, data_spec, seed=val_seed),
        train_fn=train_fn,
        val_fn=val_fn,
        run_val=run_val,
        total_steps=total_steps,
        batch_size=batch_size,
        print_every=print_every,
        save_every=val_every,
        val_every=val_every,
        val_percent=val_percent,
        done_fn=done_fn,
    )


class TokenValidation:
    def compute_accuracy(self, all_ys, all_ys_pred):
        all_ys_pred = [strip_start_and_end_tokens(y) for y in all_ys_pred]
        acc = np.mean([y == yp for y, yp in zip(all_ys, all_ys_pred)])
        return acc, all_ys_pred

    def report(self, all_ys, all_ys_pred):
        def print_toks(tag, x):
            print(tag, *[u.name for u in x])

        print_toks("Actual   :", all_ys[0])
        print_toks("Predicted:", all_ys_pred[0])


class TopKValidation:
    def compute_accuracy(self, all_ys, all_ys_pred):
        all_ys = np.array(all_ys).argmax(-1)
        if not isinstance(all_ys_pred, torch.Tensor):
            all_ys_pred = torch.stack(all_ys_pred)
        all_ys_pred = all_ys_pred.cpu().numpy()

        acc = topk_both(all_ys_pred, all_ys)

        return acc, None

    def report(self, all_ys, all_ys_pred):
        pass
